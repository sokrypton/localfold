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
 * and the score is clashes per thousand atoms.
 *
 * 🔴 AND THE FIRST VERSION OF THIS FILE WAS MIS-CALIBRATED, WHICH THE CRYSTALS
 * CAUGHT AND NOTHING ELSE COULD HAVE. It paired MolProbity's 0.4 A threshold
 * with BONDI's radii and excluded pairs within THREE bonds, and on that
 * setting a 1.5 A deposited crystal reads 3.63 and a designed protein solved
 * at 1.2 A reads 28.06 - numbers that say the instrument is wrong, not the
 * structure. Both halves were the same mistake, counting a distance that
 * chemistry fixes:
 *
 *              radii/hops  bondi 3   bondi 4   probe 3   probe 4
 *   6MRR                      3.63      0         0         0
 *   5K9P                      5.04      1.68      0         0
 *   5CAJ                      6.32      2.19      2.19      0.97
 *   1QYS                     28.06      8.86     20.68      5.91
 *   1BRS                     18.97     11.00     13.37      7.33
 *   1TIM chain A (1976)     143.85     70.59    131.02     67.38
 *
 * The dominant class at three hops is the trans-peptide O(i)...C(i+1) at about
 * 2.78 A - every protein has one per residue - which is 82 of 1TIM's 269 and
 * 11 of 5CAJ's 26. It is a bond angle. See RICHARDSON and the `hops` default
 * below for the two fixes; both are arms, so a reader can put either back.
 *
 * 🔴 WHAT IT SAYS ABOUT THE PAGE'S DEFAULT, which is the question that
 * prompted it - "are the page's defaults squashing side chains". The scale to
 * read these against: deposited crystals score 0 to 7.33 above, and
 * ALPHAFOLD 3's OWN SERVER, the one reference here that is neither this port
 * nor af3-any-model, scores **2.84** on `tools/fixtures/fold_2026_09_01_10_17.zip`.
 *
 * Five models, five targets, the page's own diffusion 25, each with an MSA
 * from api.colabfold.com - pooled clashes over pooled atoms:
 *
 *   af3 3.57   boltz2 4.08   intellifold2 5.27   protenix2 6.12   rosettafold3 9.87
 *
 * So at the page's default, with the alignment the page itself fetches, every
 * model is at the level of a deposited crystal. **The default is not the
 * cause**, and the step count is not the lever: on 5CAJ with an MSA, 25 / 50 /
 * 200 gives af3 4.75 / 5.22 / 3.32, protenix2 9.49 / 7.59 / 11.39 and
 * rosettafold3 15.19 / 15.19 / 13.77 - no direction. With a self-TEMPLATE,
 * where the fold is determined to 0.11-0.26 A, the same is true: af3 2.85 at
 * 25 steps and 5.70 at 200.
 *
 * 🔴 WHAT IS LEFT IS THE FOLD, AND A SINGLE SEQUENCE IS WHERE IT HURTS. The
 * same five models on 5CAJ from the sequence ALONE - 17-21 A, pLDDT 29-37 -
 * score 10.9 to 239.7 at 25 steps, and there more steps does help a lot
 * (protenix2 239.68 / 119.60 / 67.87). It is not the sampler getting better:
 * a wrong fold that EXPANDS has fewer overlaps, and af3's radius of gyration
 * goes 22.0 -> 24.4 A across the same arms while its score falls to 0.95,
 * below the crystal's, at 21.5 A RMSD. **A low clashscore is not a good
 * structure.** It is only evidence when read beside the RMSD.
 *
 * 🔴 AND THE CLASHES ARE NOT WHERE "SQUASHED SIDE CHAINS" WOULD PUT THEM.
 * `bySeparation` bins each clash by how far apart in the chain its two
 * residues are, and that is what separates a badly placed ROTAMER from a fold
 * that has driven two pieces of chain through each other. Over 25 arms with an
 * MSA: 170 clashes, of which **2 are local** (a residue and its neighbour) and
 * the rest are a turn of helix away or further. No model here packs a side
 * chain into its own neighbourhood; what they do is put distant things in the
 * same place.
 *
 * 🔴 AND THEY CONCENTRATE AT THE ENDS. Pooled over the same 25 arms, by decile
 * of the chain, as a ratio to what an even spread would give:
 *
 *   1.67  0.62  0.73  1.11  1.39  0.93  0.65  0.59  1.02  1.27
 *
 * The first decile is 1.7x and the last 1.3x, every middle decile but two
 * below 1. On a WRONG fold (single sequence, 2746 clashes) the same histogram
 * is flat - 0.96 at the N-terminus - so this is what a good fold's residual
 * looks like, not what a broken one does. A terminus has fewer neighbours to
 * be packed against and nothing downstream to hold it, and 5CAJ's first six
 * residues are a PRGSHM expression tag that is disordered in the crystal: the
 * model must put it somewhere and puts it on the surface.
 *
 * 🔴 AND IT IS A HANDFUL OF OUTLIERS, NOT A SYSTEMATIC SQUEEZE. The whole
 * distribution of close N...O contacts is the crystal's: median 3.10-3.16 A
 * against the crystal's 3.10 and AF3 Server's 3.13, 5th percentile 2.65-2.73
 * against 2.77 and 2.74. What differs is the tail - 4 to 13 contacts under
 * 2.5 A where the crystal has 0 and AF3 Server has 2. Compared pair by pair
 * against the crystal - fold residue i is the i-th resolved crystal residue,
 * which is what `readChain` in fold-opendde.js builds the sequence from, and
 * pairing by residue NUMBER instead reports a 14 A separation as a clash
 * because 5CAJ's chain A starts at -4 - about a third of the
 * clashes are a real interaction pulled too tight (2.82 A becoming 1.95) and
 * two thirds are a contact the crystal does not have at all.
 *
 * 🔴 A RESIDUE THE CONFORMER SET DOES NOT KNOW IS SKIPPED, AND THAT BIASES A
 * CRYSTAL LOW. `skippedResidues` counts them: 5CAJ's are 551 waters and 14
 * SELENOMETHIONINES, and the fourteen take their atoms out of the comparison
 * where a prediction's plain methionines stay in. Mapping MSE onto MET is not
 * the fix - its SE would match no bond in the conformer and read as a cage of
 * clashes, which is how the skip came to exist - so the number is reported
 * rather than hidden.
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
 * 🔴 MolProbity'S THRESHOLD NEEDS MolProbity'S RADII, AND PAIRING IT WITH
 * BONDI'S REPORTED NORMAL PEPTIDE GEOMETRY AS A CLASH. The 0.4 A cutoff was
 * calibrated by the Richardson lab against THEIR radius set (Word et al.
 * 1999), where oxygen is 1.40 A; `vanDerWaalsRadius` in
 * src/chem/geometry-tables.js is Bondi's, where it is 1.52, and that table is
 * right for what it does - it floors the conformer builder's non-bonded
 * distances - so the fix is a local set here, not an edit there.
 *
 * The 0.12 A on oxygen is the whole difference and it lands exactly on the
 * trans-peptide O(i)...C(i+1) contact, which every protein has one of per
 * residue at about 2.78 A: Bondi calls that an overlap of 0.44 and Richardson
 * 0.37. It was the single largest class of "clash" in every crystal measured
 * - 11 of 26 in 5CAJ, 82 of 269 in 1TIM - and it is a bond angle, not a
 * collision.
 */
const RICHARDSON = { C: 1.75, N: 1.55, O: 1.40, S: 1.80, P: 1.80 };

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
  // `--radii=bondi` is the control arm: the table below moves every number and
  // a reader should be able to see by how much.
  const radii = options.radii ?? RICHARDSON;
  const radius = (element) => radii[element] ?? vanDerWaalsRadius(element);
  const residues = parsePdbResidues(pdb);
  const { atoms, neighbours, skipped } = atomsAndBonds(residues, conformers);
  // 🔴 FOUR BONDS, NOT THREE, AND THE CRYSTALS ARE WHAT SAID SO. A 1-4 pair's
  // distance is set by the TORSION between them, not by whether the two atoms
  // can approach: an eclipsed rotamer puts them close and that is a strained
  // angle rather than a collision, which is bond-geometry.js's question. At
  // three the table above reads 3.63 for a 1.5 A crystal that should read
  // zero, all of it 1-4.
  const near = nearInBonds(neighbours, options.hops ?? 4);

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
      const sum = radius(a.element) + radius(b.element);
      if (distance >= sum - cutoff) continue;
      if (HYDROGEN_BONDING.has(a.element) && HYDROGEN_BONDING.has(b.element)
        && distance >= HYDROGEN_BOND_FLOOR) continue;
      found.push({
        overlap: sum - distance, distance,
        a: `${a.code}${a.number}${a.chain}:${a.name}`,
        b: `${b.code}${b.number}${b.chain}:${b.name}`,
        sidechain: !BACKBONE.has(a.name) || !BACKBONE.has(b.name),
        // 🔴 HOW FAR APART IN THE CHAIN, WHICH IS WHAT SEPARATES THE TWO
        // DIAGNOSES. A clash between a side chain and its own neighbour two
        // residues away is a ROTAMER placed wrongly on a fold that may be
        // right; a clash between residues fifty apart is two pieces of the
        // chain driven through each other, which is the FOLD being wrong and
        // no amount of side-chain repacking can fix it. The counts below are
        // the only thing here that tells the two apart, and they disagree -
        // see the table in the header.
        separation: a.chain === b.chain ? Math.abs(a.residue - b.residue) : Infinity,
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
    // Local packing against tertiary interpenetration. `local` is a residue
    // and its own immediate neighbours, `near` is a turn of helix, `tertiary`
    // is two parts of the chain that the fold has put on top of each other.
    bySeparation: {
      local: found.filter((one) => one.separation <= 1).length,
      near: found.filter((one) => one.separation > 1 && one.separation <= 4).length,
      tertiary: found.filter((one) => one.separation > 4).length,
    },
    // The unambiguous ones: nothing a hydrogen bond could explain.
    nonPolar: found.filter((one) => !one.polar).length,
    polar: found.filter((one) => one.polar).length,
    worst: found.slice(0, 5).map((one) => ({
      ...one, overlap: Number(one.overlap.toFixed(2)),
      distance: Number(one.distance.toFixed(2)),
    })),
    // Every clash, for a caller that wants to bin them itself. Off by
    // default: a broken fold has thousands and the summary is the answer.
    ...(options.all === true ? { list: found } : {}),
  };
}
