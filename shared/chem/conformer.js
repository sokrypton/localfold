/**
 * A 3D conformer for a SMILES ligand, by distance geometry.
 *
 * 🔴 WHAT THIS IS FOR, AND THEREFORE HOW GOOD IT HAS TO BE. AF3 and its
 * lineage read a ligand's reference conformer as `ref_pos`: coordinates in the
 * TOKEN'S OWN frame, used to tell the atoms of a component apart and to give
 * the atom encoder a local geometry. The sampler places the molecule itself.
 * So this needs to be a chemically sensible arrangement - right bond lengths,
 * right angles, flat rings flat, the correct enantiomer - and it does NOT need
 * to be the global energy minimum, because nothing downstream is reading an
 * energy. docs/AF3.md records this port's own idealised conformers differing
 * from the reference's CCD geometry by 0.65 A rms on intra-token distances,
 * REPORTED as a floor rather than failed on. That is the bar.
 *
 * The method is the standard one, and it is three steps:
 *
 *   1. bounds     every pair of atoms gets a lower and an upper distance, from
 *                 the bond lengths, the angles, the rings and the van der
 *                 Waals radii.
 *   2. smoothing  the triangle inequality is applied until the bounds are
 *                 consistent - without it, step 3 is handed a distance matrix
 *                 that no arrangement of points in ANY number of dimensions
 *                 can satisfy, and returns a tangle. This is the O(N^3) part.
 *   3. embedding  a distance is drawn for each pair, the metric matrix is
 *                 built and its top three eigenvectors are the coordinates,
 *                 then a few hundred steepest-descent steps clean up what is
 *                 left. This is the O(N^2)-per-step part.
 *
 * Steps 2 and 3 are the whole compute cost and both are data-parallel over
 * pairs, which is why `conformer-webgpu.js` exists beside this file. This one
 * is the reference: it is what the differential gate holds the GPU to, and it
 * is what runs when there is no device.
 */

import { adjacency } from "./smiles.js";
import { smallestRings, bondsInRings } from "./rings.js";
import {
  bondLength, idealAngle, lawOfCosines, polygonAngle, vanDerWaalsRadius,
  substituentWeight, VSEPR_STRENGTH,
} from "./geometry-tables.js";

/** How far a bond length may stray, in angstroms. */
const BOND_SLACK = 0.01;
/** ...and an angle-derived 1-3 distance, where no angle is recoverable. */
const ANGLE_SLACK = 0.04;
/** How far an open-chain bond angle may bend, in radians. */
const CHAIN_ANGLE_SLACK = (2.5 * Math.PI) / 180;
/** ...and one inside a ring, which the ring pins. */
const RING_ANGLE_SLACK = (1.2 * Math.PI) / 180;
/**
 * How hard planarity is enforced against the distance bounds.
 *
 * 🔴 IT HAS TO OUTWEIGH A BOUND THE PUCKER WOULD OTHERWISE SATISFY, and it
 * must not be so large that it flattens a ring the bounds say is a chair.
 * `isPlanarRing` is what decides WHICH rings hear this at all, so the weight
 * only has to win inside a genuinely flat ring - at 1.0 benzene still came out
 * 0.2 A puckered and at 10 it is under 0.01, with cyclohexane's chair
 * untouched because it is never in the list.
 */
const PLANARITY_WEIGHT = 0.5;
/** ...and how hard a linear centre is held straight; see `linearTriples`. */
const LINEARITY_WEIGHT = 0.5;
/** How far an across-the-ring distance may stray; a real ring is not regular. */
const RING_SLACK = 0.10;
/** Two atoms four bonds apart may not come closer than this share of their vdW sum. */
const CLASH_FRACTION = 0.8;

/**
 * Lower and upper distance bounds for every pair.
 *
 * Returned as two flat `n * n` Float64Arrays, which is the layout the GPU
 * path wants and costs the CPU path nothing.
 *
 * @param {{atoms: object[], bonds: object[]}} graph
 * @returns {{lower: Float64Array, upper: Float64Array, n: number, rings: number[][]}}
 */
export function distanceBounds(graph) {
  const n = graph.atoms.length;
  const lower = new Float64Array(n * n);
  const upper = new Float64Array(n * n).fill(1000);
  const lists = adjacency(graph);
  const rings = smallestRings(graph);
  const inRing = bondsInRings(graph, rings);

  for (let i = 0; i < n; i += 1) { lower[i * n + i] = 0; upper[i * n + i] = 0; }
  // How many bonds apart each pair is, which decides which rules may speak.
  const hops = topologicalDistance(lists, n);

  // 1-2: the bond itself, which is the tightest thing known.
  const bondDistance = new Map();
  graph.bonds.forEach((bond, index) => {
    const a = graph.atoms[bond.from].symbol;
    const b = graph.atoms[bond.to].symbol;
    const order = bond.geometryOrder ?? bond.order;
    const length = bondLength(a, b, order, bond.aromatic === true,
                              graph.atoms[bond.from].charge, graph.atoms[bond.to].charge);
    setBound(lower, upper, n, bond.from, bond.to,
             length - BOND_SLACK, length + BOND_SLACK);
    bondDistance.set(key(bond.from, bond.to), length);
    void index; void inRing;
  });

  // 1-3: two bonds and the angle between them.
  //
  // 🔴 THE SIGMA COUNT INCLUDES HYDROGENS, which this port drops before the
  // featuriser sees them but must count here: a carbonyl carbon has two heavy
  // neighbours and is trigonal at 120 degrees, and counting heavy atoms alone
  // makes it linear.
  const sigma = graph.atoms.map((atom, index) =>
    lists[index].length + (atom.hydrogens ?? 0));
  // ...and whether the atom carries a pi bond, which is the other half of what
  // names its shape: `true` for one, `"two"` for a triple or two doubles.
  const pi = graph.atoms.map((atom, index) => {
    if (atom.aromatic) return true;
    let extra = 0;
    for (const step of lists[index]) {
      const order = graph.bonds[step.bond].geometryOrder ?? graph.bonds[step.bond].order;
      if (order >= 3) extra += 2;
      else if (order > 1) extra += 1;
    }
    return extra >= 2 ? "two" : extra === 1 ? true : false;
  });
  //
  // 🔴 AND THE ESTIMATES FOR ONE PAIR ARE COLLECTED BEFORE ANY BOUND IS SET,
  // BECAUSE A PAIR CAN HAVE TWO. Two atoms with two common neighbours are the
  // diagonal of a four-ring, and each path gives its own answer: penicillin's
  // beta-lactam is reached through a carbon (two C-C bonds, 2.15 A) and
  // through a nitrogen (C-N and an amide N-C, 2.01 A). Those are two
  // ESTIMATES OF ONE DISTANCE, and intersecting them - which is what setting
  // each bound as it is computed does - produces a lower above an upper and a
  // molecule with no valid geometry at all. The true diagonal lies between
  // them, so the bound must SPAN them. That was penicillin's last
  // contradiction, and it is the only place in this file where the order of
  // two rules mattered.
  // Precomputed per ring, because the whole ring is solved at once.
  const ringAngles = new Map();
  for (const ring of rings) {
    if (ring.length < 3 || ring.length > 6) continue;
    const solved = cyclicRingAngles(ring, bondDistance);
    if (solved === null) continue;
    for (const [atom, value] of solved) {
      const at = `${ring.length}:${atom}`;
      if (!ringAngles.has(at)) ringAngles.set(at, value);
    }
  }

  const angleEstimates = new Map();
  for (let centre = 0; centre < n; centre += 1) {
    const neighbours = lists[centre];
    for (let a = 0; a < neighbours.length; a += 1) {
      for (let b = a + 1; b < neighbours.length; b += 1) {
        const first = neighbours[a].atom;
        const second = neighbours[b].atom;
        const da = bondDistance.get(key(centre, first));
        const db = bondDistance.get(key(centre, second));
        if (da === undefined || db === undefined) continue;
        // A ring smaller than six forces its own interior angle, whatever the
        // hybridisation would like: cyclopropane's carbons are at 60 degrees
        // and no amount of sp3 wanting 109 changes that.
        const shared = smallestSharedRing(rings, centre, first, second);
        // 🔴 AND THE ANGLE LEANS TOWARDS WHICHEVER SUBSTITUENT IS BIGGER. See
        // `substituentWeight`: a double bond pushes its neighbours away, so
        // the angle to it opens and the angle between the remaining single
        // bonds closes. A ring's interior angle is fixed by the ring and hears
        // none of this.
        const solved = shared === null ? undefined
          : ringAngles.get(`${shared}:${centre}`);
        const angle = shared !== null && shared <= 5
          ? (solved ?? polygonAngle(shared))
          : vseprAngle(graph, lists, sigma, pi, centre, first, second);
        const distance = lawOfCosines(da, db, angle);
        const pair = key(first, second);
        const seen = angleEstimates.get(pair);
        if (seen === undefined) {
          angleEstimates.set(pair, { first, second, low: distance, high: distance,
                                     shared, da, db });
        } else {
          seen.low = Math.min(seen.low, distance);
          seen.high = Math.max(seen.high, distance);
          seen.shared = seen.shared === null ? shared
            : shared === null ? seen.shared : Math.min(seen.shared, shared);
        }
      }
    }
  }
  for (const estimate of angleEstimates.values()) {
    // 🔴 THE TOLERANCE IS AN ANGLE, CONVERTED TO A DISTANCE - NOT A DISTANCE.
    // A 1-3 distance is `sqrt(a^2 + b^2 - 2ab cos(theta))`, whose sensitivity
    // to theta goes as sin(theta), so near 180 degrees it barely moves at all:
    // a fixed +/-0.04 A on a nitrile's C-C-N, where the distance is already
    // the maximum two bonds can reach, is +/-14 DEGREES of angle - and the
    // optimiser spends all of it, bending an sp carbon that is linear to
    // within a degree in reality. Measured: CC#N came out at 166 degrees and
    // allene at 163, both with their bounds perfectly satisfied.
    //
    // Converting the other way makes the tolerance mean the same thing at
    // every angle, which is what it was always supposed to mean.
    const tolerance = estimate.shared !== null ? RING_ANGLE_SLACK : CHAIN_ANGLE_SLACK;
    const spread = (distance, sign) => {
      // The angle this estimate implies, widened by the tolerance and turned
      // back into a distance through the same law of cosines.
      const a = estimate.da;
      const b = estimate.db;
      if (a === undefined || b === undefined) return distance + sign * ANGLE_SLACK;
      const cosine = (a * a + b * b - distance * distance) / (2 * a * b);
      const angle = Math.acos(Math.max(-1, Math.min(1, cosine)));
      const widened = Math.max(0, Math.min(Math.PI, angle + sign * tolerance));
      return lawOfCosines(a, b, widened);
    };
    // 🔴 A LINEAR CENTRE IS A SPECIAL CASE AND WIDENING IS THE WRONG ANSWER TO
    // IT. At 180 degrees the 1-3 distance IS the sum of the two bonds, and
    // each bond is free inside its own +/-0.01 A - so a 1-3 bound pinned to
    // the sum of the IDEAL lengths forces a bend the moment both bonds settle
    // high, which is how a nitrile came out at 165 degrees with every bound
    // satisfied. The tempting repair is to widen the 1-3 bound by the bonds'
    // slack; measured, that made the whole corpus worse (mean angle 2.90 ->
    // 3.54 degrees, and the nitrile itself 14.3 -> 20.2) because it loosens
    // every OTHER angle to fix one.
    //
    // What is actually true is that the distance TRACKS the bonds rather than
    // having a tolerance of its own. Stating that - the bound is the sum of
    // the two bond ranges, with no angular slack at all - pins the centre
    // collinear and still lets the bonds breathe.
    const linear = estimate.high > 0.999 * ((estimate.da ?? 0) + (estimate.db ?? 0));
    if (linear && estimate.da !== undefined) {
      setBound(lower, upper, n, estimate.first, estimate.second,
               estimate.da + estimate.db - 2 * BOND_SLACK,
               estimate.da + estimate.db + 2 * BOND_SLACK);
    } else {
      setBound(lower, upper, n, estimate.first, estimate.second,
               spread(estimate.low, -1), spread(estimate.high, 1));
    }
  }

  // 🔴 A FLAT RING IS FLAT AND THAT FIXES EVERY DISTANCE IN IT. An aromatic
  // ring is planar and regular to a good approximation, so the 1-4 distance
  // across a benzene is not "somewhere between cis and trans" - it is 2.78 A.
  // Left to the generic 1-4 rule the ring can pucker, and a puckered benzene
  // is the most visible thing a wrong conformer produces.
  for (const ring of rings) {
    // 🔴 AND ONLY A SMALL RING. A regular polygon is a fair model of a benzene
    // and a nonsense model of porphine's 16-membered macrocycle, which is not
    // convex, not equilateral and not remotely circular - applied there it
    // produced 56 contradictory bounds. Past about seven atoms a ring's shape
    // is decided by the bonds and angles around it, which the rules above
    // already state, and this one has nothing to add.
    if (ring.length > 7) continue;
    if (!isPlanarRing(graph, ring)) continue;
    const radius = meanRingRadius(ring, bondDistance);
    if (radius === null) continue;
    for (let a = 0; a < ring.length; a += 1) {
      for (let b = a + 1; b < ring.length; b += 1) {
        const separation = Math.min(b - a, ring.length - (b - a));
        // 🔴 SEPARATION 2 IS A 1-3 PAIR AND THE ANGLE RULE ALREADY KNOWS IT
        // EXACTLY, from the two real bond lengths. This rule models the ring
        // as a REGULAR polygon of the mean bond length, which a purine is not
        // - its C-N bonds are 1.33 and its C-C 1.40 - so the two rules
        // disagree by a few hundredths on every fused ring, and since one
        // sets the lower bound and the other the upper they cross. Caffeine
        // reported 4 contradictions and ATP 13, all of them separation-2
        // pairs in the fused ring. The angle rule owns those; this one speaks
        // only about distances across the ring that nothing else reaches.
        if (separation < 3) continue;
        const angle = (2 * Math.PI * separation) / ring.length;
        const distance = 2 * radius * Math.sin(angle / 2);
        // ...and with a wider tolerance than an angle, because a regular
        // polygon IS an approximation for a ring of unequal bonds.
        setBound(lower, upper, n, ring[a], ring[b],
                 distance - RING_SLACK, distance + RING_SLACK);
      }
    }
  }

  // 1-4: free rotation spans cis to trans, unless the middle bond cannot turn.
  for (const bond of graph.bonds) {
    const rotatable = (bond.geometryOrder ?? bond.order) < 1.5
      && !ringBond(rings, bond.from, bond.to);
    for (const first of lists[bond.from]) {
      if (first.atom === bond.to) continue;
      for (const second of lists[bond.to]) {
        if (second.atom === bond.from || second.atom === first.atom) continue;
        const cis = distanceAt(graph, lists, bondDistance, sigma, pi, rings,
                               first.atom, bond.from, bond.to, second.atom, 0);
        const trans = distanceAt(graph, lists, bondDistance, sigma, pi, rings,
                                 first.atom, bond.from, bond.to, second.atom, Math.PI);
        if (cis === null || trans === null) continue;
        if (rotatable) {
          setBound(lower, upper, n, first.atom, second.atom,
                   Math.min(cis, trans) - ANGLE_SLACK, Math.max(cis, trans) + ANGLE_SLACK);
        } else if (!ringBond(rings, bond.from, bond.to)) {
          // A double bond holds its substituents in a plane; which side is a
          // question for the stereo pass, so both are allowed but the
          // in-between is not - expressed as the loosest bound that still
          // forbids a twisted double bond from stretching past trans.
          setBound(lower, upper, n, first.atom, second.atom,
                   Math.min(cis, trans) - ANGLE_SLACK, Math.max(cis, trans) + ANGLE_SLACK);
        }
      }
    }
  }

  // 🔴 AND THE CLASH FLOOR GOES LAST, ON THE PAIRS NOTHING ELSE REACHED. Put
  // first it contradicts every bond in the molecule: two bonded carbons are
  // 1.52 A apart and 0.8 times their van der Waals sum is 2.72, so the lower
  // bound lands above the upper and the whole molecule is unsatisfiable -
  // glycerol reported 10 contradictions, ATP 86, before this moved. A van der
  // Waals radius describes two atoms that are NOT bonded to each other, and
  // saying so means applying it only from four bonds apart, where no bond,
  // angle or ring geometry has an opinion.
  //
  // It is also capped by the upper bound it meets: a fused ring system can
  // hold two atoms four bonds apart closer than their radii would like, and
  // there the ring is right and the radii are a generalisation.
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (hops[i * n + j] < 4) continue;
      const clash = CLASH_FRACTION * (vanDerWaalsRadius(graph.atoms[i].symbol)
        + vanDerWaalsRadius(graph.atoms[j].symbol));
      const capped = Math.min(clash, upper[i * n + j]);
      if (capped > lower[i * n + j]) {
        lower[i * n + j] = capped;
        lower[j * n + i] = capped;
      }
    }
  }

  // ...and a pair that still has no floor at all gets a token one, or the
  // embedding is free to put two atoms in the same place, which satisfies
  // every upper bound at once and is the cheapest way to score well.
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (lower[i * n + j] > 0.1) continue;
      const floor = Math.min(1.0, upper[i * n + j] * 0.5);
      lower[i * n + j] = floor;
      lower[j * n + i] = floor;
    }
  }

  return { lower, upper, n, rings, sigma, pi };
}

/** How many bonds apart every pair is, by breadth-first search. */
function topologicalDistance(lists, n) {
  const hops = new Int32Array(n * n).fill(1e6);
  for (let start = 0; start < n; start += 1) {
    hops[start * n + start] = 0;
    const queue = [start];
    for (let head = 0; head < queue.length; head += 1) {
      const atom = queue[head];
      const next = hops[start * n + atom] + 1;
      for (const step of lists[atom]) {
        if (hops[start * n + step.atom] <= next) continue;
        hops[start * n + step.atom] = next;
        queue.push(step.atom);
      }
    }
  }
  return hops;
}

const key = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

function setBound(lower, upper, n, i, j, low, high) {
  // 🔴 A TIGHTER BOUND ALWAYS WINS. Rules overlap - a 1-3 pair inside a flat
  // ring is reached by the angle rule and by the ring rule - and the later one
  // must not loosen what the earlier one knew. Taking the max of the lowers
  // and the min of the uppers is what makes the order of these passes not
  // matter, which is worth more than any one of them being first.
  const at = i * n + j;
  const back = j * n + i;
  const newLow = Math.max(lower[at], low);
  const newHigh = Math.min(upper[at], high);
  lower[at] = newLow; lower[back] = newLow;
  upper[at] = newHigh; upper[back] = newHigh;
}

function ringBond(rings, a, b) {
  return rings.some((ring) => {
    const i = ring.indexOf(a);
    if (i < 0) return false;
    return ring[(i + 1) % ring.length] === b
      || ring[(i - 1 + ring.length) % ring.length] === b;
  });
}

/**
 * The interior angles of a flat ring whose sides are not all equal.
 *
 * 🔴 A REGULAR POLYGON IS THE WRONG MODEL FOR A HETEROAROMATIC RING AND
 * THIOPHENE IS THE PROOF. Its S-C bonds are 1.71 A and its C-C bonds 1.37, so
 * the five interior angles are nothing like the pentagon's 108: the real
 * C-S-C is about 92 and the carbons open past 111 to compensate. Using 108 for
 * all five was the worst angle left in the corpus at 5.8 degrees, and it is
 * wrong in the same direction for furan, pyrrole, imidazole and every other
 * five-ring with a heteroatom - which is most of the ones a ligand has.
 *
 * A planar ring with given side lengths is a CYCLIC POLYGON, inscribed in some
 * circle, and that circle is the only unknown: each side `b` subtends
 * `2 * asin(b / 2R)` at the centre and the subtended angles must sum to a full
 * turn. One bisection on R solves it, and then each interior angle follows.
 * Exact for a planar ring, and it degenerates to the regular polygon when the
 * sides happen to be equal, so nothing that was right becomes wrong.
 *
 * @returns {Map<string, number>|null} keyed by the vertex's atom index
 */
function cyclicRingAngles(ring, bondDistance) {
  const sides = [];
  for (let index = 0; index < ring.length; index += 1) {
    const length = bondDistance.get(key(ring[index], ring[(index + 1) % ring.length]));
    if (length === undefined) return null;
    sides.push(length);
  }
  const longest = Math.max(...sides);
  // The circle must at least contain the longest side as a chord.
  let low = longest / 2;
  let high = longest * ring.length;
  const turn = (radius) => sides.reduce(
    (total, side) => total + 2 * Math.asin(Math.min(1, side / (2 * radius))), 0);
  // 🔴 THE TURN DECREASES AS THE RADIUS GROWS, so the bisection looks for
  // where it crosses 2*pi from ABOVE. At the smallest legal radius the sides
  // wrap more than once round; at a large one they barely bend.
  if (turn(high) > 2 * Math.PI) return null;      // no circle fits these sides
  for (let step = 0; step < 60; step += 1) {
    const middle = (low + high) / 2;
    if (turn(middle) > 2 * Math.PI) low = middle; else high = middle;
  }
  const radius = (low + high) / 2;
  const half = sides.map((side) => Math.asin(Math.min(1, side / (2 * radius))));
  const angles = new Map();
  for (let index = 0; index < ring.length; index += 1) {
    // The interior angle at vertex i sits between sides i-1 and i.
    const before = half[(index - 1 + ring.length) % ring.length];
    const after = half[index];
    angles.set(String(ring[index]), Math.PI - before - after);
  }
  return angles;
}

/** The smallest ring holding all three atoms consecutively, or null. */
function smallestSharedRing(rings, centre, first, second) {
  let best = null;
  for (const ring of rings) {
    if (!ring.includes(centre) || !ring.includes(first) || !ring.includes(second)) continue;
    if (best === null || ring.length < best) best = ring.length;
  }
  return best;
}

/** True when every atom of the ring is aromatic or sp2, so the ring is flat. */
function isPlanarRing(graph, ring) {
  return ring.every((atom) => {
    if (graph.atoms[atom].aromatic) return true;
    const sigma = graph.bonds.filter((bond) =>
      bond.from === atom || bond.to === atom).length + (graph.atoms[atom].hydrogens ?? 0);
    const hasDouble = graph.bonds.some((bond) =>
      (bond.from === atom || bond.to === atom) && (bond.geometryOrder ?? bond.order) >= 2);
    return sigma === 3 && hasDouble;
  });
}

/** The circumradius implied by the ring's own bond lengths. */
function meanRingRadius(ring, bondDistance) {
  let total = 0;
  for (let index = 0; index < ring.length; index += 1) {
    const length = bondDistance.get(key(ring[index], ring[(index + 1) % ring.length]));
    if (length === undefined) return null;
    total += length;
  }
  const side = total / ring.length;
  return side / (2 * Math.sin(Math.PI / ring.length));
}

/**
 * The i-l distance across the torsion i-j-k-l at a given dihedral.
 *
 * Built by placing the four atoms explicitly rather than by a closed form,
 * because the closed form for a general torsion is long enough to get wrong
 * and this runs once per 1-4 pair at build time.
 */
function distanceAt(graph, lists, bondDistance, sigma, pi, rings, i, j, k, l, dihedral) {
  const dij = bondDistance.get(key(i, j));
  const djk = bondDistance.get(key(j, k));
  const dkl = bondDistance.get(key(k, l));
  if (dij === undefined || djk === undefined || dkl === undefined) return null;
  const angleIJK = ringAwareAngle(graph, sigma, pi, rings, j, i, k);
  const angleJKL = ringAwareAngle(graph, sigma, pi, rings, k, j, l);
  void lists;

  // 🔴 j AT THE ORIGIN AND k ALONG +x, AND THE TWO OUTER ATOMS ARE PLACED FROM
  // THE ANGLE ITSELF, NOT FROM ITS SUPPLEMENT. The first version used
  // `PI - angle` for both, which mirrors i and l through the x axis and puts
  // them on the SAME side instead of opposite ones: an sp3 1-4 pair came out
  // 0.51 A apart where it should be 2.53, so every torsion's lower bound was
  // roughly a bond length. Alanine ended up with `0C-4O: [1.19, 2.58]` - a
  // lower bound below any real bond - and the bounds as a set had no
  // three-dimensional solution at all, which the pairwise contradiction check
  // cannot see because no single pair is inconsistent. It showed up as an
  // optimiser that ran its full step budget and stopped at an error of 0.1
  // with every bound violated.
  //
  // The angle at j is between j->i and j->k, and j->k is +x, so i sits at
  // `angleIJK` from +x. The angle at k is between k->j and k->l, and k->j is
  // -x, so l leans back towards j by the same rule - which is the minus sign
  // below, with the dihedral turning it out of the plane.
  const kp = [djk, 0, 0];
  const ip = [dij * Math.cos(angleIJK), dij * Math.sin(angleIJK), 0];
  const lx = kp[0] - dkl * Math.cos(angleJKL);
  const radial = dkl * Math.sin(angleJKL);
  const lp = [lx, radial * Math.cos(dihedral), radial * Math.sin(dihedral)];
  return Math.hypot(ip[0] - lp[0], ip[1] - lp[1], ip[2] - lp[2]);
}

/**
 * The ideal angle at `centre`, leaned towards the heavier substituent.
 *
 * The shift is measured against the MEAN weight of the centre's substituents,
 * so a centre whose substituents are all alike gets exactly the base angle and
 * this whole rule vanishes - which is what makes it safe to add underneath an
 * existing model rather than beside it.
 */
function vseprAngle(graph, lists, sigma, pi, centre, first, second) {
  const base = idealAngle(sigma[centre], graph.atoms[centre].symbol, pi[centre]);
  const neighbours = lists[centre];
  if (neighbours.length < 3) return base;
  const weightOf = (step) => substituentWeight(
    graph.bonds[step.bond].geometryOrder ?? graph.bonds[step.bond].order,
    graph.bonds[step.bond].aromatic === true);
  let total = 0;
  for (const step of neighbours) total += weightOf(step);
  const mean = total / neighbours.length;
  const pick = (atom) => neighbours.find((step) => step.atom === atom);
  const a = pick(first);
  const b = pick(second);
  if (a === undefined || b === undefined) return base;
  const shift = (weightOf(a) + weightOf(b) - 2 * mean) * VSEPR_STRENGTH;
  // Clamped, because a centre with one very heavy substituent would otherwise
  // be pushed past a geometry that exists at all.
  const degrees = Math.max(60, Math.min(180, (base * 180) / Math.PI + shift));
  return (degrees * Math.PI) / 180;
}

function ringAwareAngle(graph, sigma, pi, rings, centre, first, second) {
  const shared = smallestSharedRing(rings, centre, first, second);
  return shared !== null && shared <= 5
    ? polygonAngle(shared)
    : idealAngle(sigma[centre], graph.atoms[centre].symbol, pi[centre]);
}

/**
 * Make the bounds consistent with the triangle inequality.
 *
 * 🔴 WITHOUT THIS THE EMBEDDING IS SOLVING AN IMPOSSIBLE PROBLEM AND RETURNS A
 * TANGLE. The bounds are built from local rules that know nothing about each
 * other, so they routinely allow a distance that no arrangement of points
 * admits: three atoms whose pairwise uppers are 1.5, 1.5 and 5 describe a
 * triangle with one side longer than the other two together. Smoothing walks
 * that back - the upper bounds by a shortest-path (Floyd-Warshall), the lower
 * ones by the matching rule that a lower cannot exceed the shortest way round.
 *
 * O(N^3), and the reason `conformer-webgpu.js` exists: at a 60-atom ligand
 * that is 216,000 relaxations, and it is the same inner loop for every one.
 */
export function smoothBounds(bounds) {
  const { lower, upper, n } = bounds;
  // Upper: the shortest path through any intermediate atom.
  for (let k = 0; k < n; k += 1) {
    for (let i = 0; i < n; i += 1) {
      const ik = upper[i * n + k];
      for (let j = i + 1; j < n; j += 1) {
        const through = ik + upper[k * n + j];
        if (through < upper[i * n + j]) {
          upper[i * n + j] = through;
          upper[j * n + i] = through;
        }
      }
    }
  }
  // Lower: two atoms cannot be closer than one leg minus the other.
  for (let k = 0; k < n; k += 1) {
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const a = lower[i * n + k] - upper[k * n + j];
        const b = lower[k * n + j] - upper[i * n + k];
        const best = Math.max(a, b);
        if (best > lower[i * n + j]) {
          lower[i * n + j] = best;
          lower[j * n + i] = best;
        }
      }
    }
  }
  // 🔴 AND AN INCONSISTENT PAIR IS REPORTED, NOT CLAMPED SILENTLY. A lower
  // above its upper means the rules contradict each other, which is a bug in
  // the rules and not something to paper over with a swap - it would place the
  // atom somewhere arbitrary and the conformer would look merely odd.
  let contradictions = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (lower[i * n + j] > upper[i * n + j] + 1e-9) contradictions += 1;
    }
  }
  return contradictions;
}

/**
 * Groups of four atoms that must be coplanar, as `[a, b, c, d]` quadruples.
 *
 * 🔴 DISTANCE BOUNDS CANNOT SAY "FLAT" AND THIS IS THE PROOF. Benzene's
 * bounds pin every bond at 1.43, every 1-3 at 2.48 and every para pair at
 * 2.76-2.96 - and a ring puckered 0.44 A out of plane satisfies all fifteen
 * of them, which is exactly what the embedder returned, with a bounds error of
 * 2.6e-9. It was not failing to solve the problem; the problem did not say
 * what was wanted. Tightening the para bound would only trade one artefact
 * for another, because a FUSED ring is not regular and its across-ring
 * distance genuinely varies.
 *
 * So planarity is stated directly, the way chirality is: the signed volume of
 * four coplanar atoms is zero, and `refineCoordinates` drives it there. Two
 * kinds of group qualify - an aromatic or conjugated RING, taken as
 * consecutive quadruples round it, and an sp2 CENTRE with its three
 * neighbours, which is what makes an amide and a carboxylate flat.
 */
export function planarQuadruples(graph, rings) {
  const quadruples = [];
  for (const ring of rings) {
    if (ring.length < 4 || ring.length > 7) continue;      // a 3-ring is flat anyway
    if (!isPlanarRing(graph, ring)) continue;
    for (let index = 0; index < ring.length; index += 1) {
      quadruples.push([
        ring[index],
        ring[(index + 1) % ring.length],
        ring[(index + 2) % ring.length],
        ring[(index + 3) % ring.length],
      ]);
    }
  }
  // An sp2 atom and its three neighbours: the carbonyl, the amide, the
  // carboxylate, the aromatic substituent. Without it an amide nitrogen
  // pyramidalises and every angle around it is a few degrees out.
  const lists = adjacency(graph);
  graph.atoms.forEach((atom, index) => {
    const neighbours = lists[index].map((step) => step.atom);
    const sigma = neighbours.length + (atom.hydrogens ?? 0);
    if (sigma !== 3 || neighbours.length !== 3) return;
    // 🔴 SULFUR AND PHOSPHORUS ARE NOT PLANAR EVEN WITH A DOUBLE BOND, and
    // this rule said they were. It is `idealAngle`'s mistake in a second
    // place: a sulfoxide keeps a stereochemically active lone pair, so it is
    // pyramidal at about 106 degrees - which is why a sulfoxide can be a
    // stereocentre at all. Flattened here, dimethyl sulfoxide came out with
    // all three angles at 120 against RDKit's 95.8 and 107.5, the worst angle
    // in the corpus at 16.4 degrees, and the angle table's own 106 could not
    // win against a planarity term pulling the other way.
    if (["S", "SE", "P", "AS"].includes(atom.symbol)) return;
    const hasPi = atom.aromatic || lists[index].some((step) =>
      (graph.bonds[step.bond].geometryOrder ?? graph.bonds[step.bond].order) > 1);
    if (!hasPi) return;
    quadruples.push([neighbours[0], neighbours[1], neighbours[2], index]);
  });
  return quadruples;
}

/**
 * Triples that must be collinear, as `[first, centre, second]`.
 *
 * 🔴 DISTANCE BOUNDS CANNOT SAY "STRAIGHT" ANY MORE THAN THEY COULD SAY
 * "FLAT", AND FOR A SHARPER REASON. A 1-3 distance is
 * `sqrt(a^2 + b^2 - 2ab cos(theta))`, whose sensitivity to theta goes as
 * sin(theta) - which is ZERO at 180 degrees. So near a linear centre the
 * distance carries almost no information about the angle: on a nitrile, a
 * bound of +/-0.02 A around the ideal still admits 166 degrees, and the two
 * bonds' own +/-0.01 A tolerances alone are enough to force a bend. Measured
 * at every repair tried from the distance side - angular tolerance, inheriting
 * the bond slack, tracking the bond sum exactly - the nitrile stayed between
 * 14 and 20 degrees off, and widening to fix it made the whole corpus worse.
 *
 * So collinearity is stated directly, exactly as planarity and chirality are.
 * The cross product of the two bond vectors is zero when they are opposite,
 * and `refineCoordinates` drives its square down.
 */
export function linearTriples(graph, sigma, pi) {
  const lists = adjacency(graph);
  const triples = [];
  graph.atoms.forEach((atom, index) => {
    const neighbours = lists[index].map((step) => step.atom);
    if (neighbours.length !== 2) return;
    const angle = idealAngle(sigma[index], atom.symbol, pi[index]);
    if (angle < Math.PI - 0.01) return;
    triples.push([neighbours[0], index, neighbours[1]]);
  });
  return triples;
}

/**
 * A seeded generator, so a conformer is the same conformer twice.
 *
 * 🔴 DISTANCE GEOMETRY IS RANDOMISED AND THAT MUST NOT REACH THE FOLD.
 * `ref_pos` is an input feature: a ligand whose reference conformer differs
 * between two runs makes the whole fold irreproducible, and the difference
 * would show up as a seed effect somewhere else entirely. The seed is derived
 * from the SMILES in `component.js`, so the same string always gives the same
 * coordinates on any machine.
 */
function generator(seed) {
  let state = (seed >>> 0) || 1;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >> 17;
    state ^= state << 5; state >>>= 0;
    return state / 4294967296;
  };
}

/**
 * Coordinates for the graph, from smoothed bounds.
 *
 * @param {{lower: Float64Array, upper: Float64Array, n: number}} bounds
 * @param {number} seed
 * @returns {Float64Array} `3n`, laid out x,y,z per atom
 */
export function embedBounds(bounds, seed) {
  const { lower, upper, n } = bounds;
  const random = generator(seed);
  if (n === 1) return new Float64Array(3);

  // A distance drawn for every pair, inside its bounds.
  const chosen = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const low = lower[i * n + j];
      const high = Math.min(upper[i * n + j], low + 50);
      const value = low + (high - low) * random();
      chosen[i * n + j] = value;
      chosen[j * n + i] = value;
    }
  }

  // 🔴 THE METRIC MATRIX NEEDS DISTANCES TO THE CENTROID AND THEY ARE NOT
  // GIVEN. They are recovered from the pairwise distances alone, which is what
  // makes this classical scaling rather than a fit: the centroid's distance to
  // atom i is the mean of the squares to every other atom, less half the mean
  // of all the squares. Getting that second term wrong tilts the whole
  // molecule and is invisible in any single distance.
  const squared = (i, j) => chosen[i * n + j] * chosen[i * n + j];
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) total += squared(i, j);
  }
  total /= 2 * n * n;
  const toCentre = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    let row = 0;
    for (let j = 0; j < n; j += 1) row += squared(i, j);
    toCentre[i] = row / n - total;
  }

  const metric = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      metric[i * n + j] = (toCentre[i] + toCentre[j] - squared(i, j)) / 2;
    }
  }

  // Top three eigenvectors by power iteration with deflation. Three of them,
  // because three is the dimension the answer lives in.
  //
  // 🔴 SHIFTED POSITIVE FIRST, OR AN ENTIRE AXIS COLLAPSES AND THE MOLECULE
  // COMES OUT FLAT. Power iteration converges to the largest eigenvalue by
  // MAGNITUDE, and a metric matrix built from distances drawn at random inside
  // their bounds is NOT positive semi-definite - those distances need not be
  // realisable in three dimensions, or in any number. So the iteration
  // happily returns a large NEGATIVE eigenvalue, `sqrt(max(value, 0))` makes
  // its scale zero, and that coordinate axis is identically zero for every
  // atom. Measured: ATP's y spread was 0.000 and biotin's z was 0.000 - both
  // molecules embedded PLANAR, which is why their signed volumes were exactly
  // zero and every chiral centre read as neither hand.
  //
  // Adding `shift * I` moves every eigenvalue up without moving a single
  // eigenVECTOR, so the iteration finds the most POSITIVE direction and the
  // original eigenvalue is recovered by subtracting the shift again.
  // Gershgorin's bound is the cheapest shift that is certainly large enough.
  let shift = 0;
  for (let i = 0; i < n; i += 1) {
    let row = 0;
    for (let j = 0; j < n; j += 1) row += Math.abs(metric[i * n + j]);
    if (row > shift) shift = row;
  }
  const coordinates = new Float64Array(3 * n);
  const work = new Float64Array(n * n);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      work[i * n + j] = metric[i * n + j] + (i === j ? shift : 0);
    }
  }
  for (let axis = 0; axis < 3; axis += 1) {
    const { vector, value } = dominantEigenvector(work, n, random);
    const original = value - shift;
    // 🔴 AND A STILL-NEGATIVE AXIS GETS NOISE RATHER THAN ZEROS. It means the
    // drawn distances genuinely do not fit in the dimensions left, which is
    // normal - the refinement's whole job is to repair that. What is not
    // recoverable is a flat start: every chiral centre sits at a volume of
    // exactly zero, so the term that should push it to one hand feels no
    // gradient at all and the molecule stays flat.
    const scale = original > 0 ? Math.sqrt(original) : 0;
    for (let i = 0; i < n; i += 1) {
      coordinates[i * 3 + axis] = scale > 0
        ? vector[i] * scale : (random() - 0.5) * 0.5;
    }
    // Deflate, so the next pass finds the next axis rather than this one again.
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) work[i * n + j] -= value * vector[i] * vector[j];
    }
  }
  return coordinates;
}

function dominantEigenvector(matrix, n, random) {
  let vector = new Float64Array(n);
  for (let i = 0; i < n; i += 1) vector[i] = random() - 0.5;
  normalise(vector);
  let value = 0;
  for (let step = 0; step < 200; step += 1) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      let sum = 0;
      for (let j = 0; j < n; j += 1) sum += matrix[i * n + j] * vector[j];
      next[i] = sum;
    }
    const length = Math.hypot(...next);
    if (length < 1e-12) break;
    for (let i = 0; i < n; i += 1) next[i] /= length;
    const change = next.reduce((most, x, i) => Math.max(most, Math.abs(x - vector[i])), 0);
    vector = next;
    value = length;
    if (change < 1e-10) break;
  }
  // The Rayleigh quotient, because the power iteration's length is the
  // magnitude and loses the SIGN - and a deflation with the wrong sign adds
  // the component back instead of removing it.
  let quotient = 0;
  for (let i = 0; i < n; i += 1) {
    let sum = 0;
    for (let j = 0; j < n; j += 1) sum += matrix[i * n + j] * vector[j];
    quotient += vector[i] * sum;
  }
  return { vector, value: quotient };
}

function normalise(vector) {
  const length = Math.hypot(...vector);
  if (length > 0) for (let i = 0; i < vector.length; i += 1) vector[i] /= length;
}

/**
 * Pull every violated pair straight onto its bound, repeatedly.
 *
 * 🔴 STEEPEST DESCENT IS THE WRONG SOLVER FOR A DISTANCE CONSTRAINT AND THE
 * NUMBERS SAY SO. Measured over 24 random starts, the gradient refinement
 * alone reached a satisfied set of bounds for glycerol 23 times, for a CF3
 * group **3 times**, and for ATP **not once** - its best of 24 was an error of
 * 0.201. That is not a tolerance to widen: a CF3 carbon was coming out at
 * F-C-F 161 degrees and F-C-C 180, which is a planar carbon, and the bounds
 * pinning it at 109.47 were perfectly correct and simply not being met.
 *
 * A violated distance has an exact, local repair - move the two atoms along
 * the line between them until they are the right distance apart - and applying
 * that pair by pair is Gauss-Seidel on the constraint set. It is the same idea
 * as SHAKE in molecular dynamics and as position-based dynamics in a physics
 * engine. It has no step size to choose and no line search to stall, and each
 * sweep is O(N^2) with a tiny constant.
 *
 * It cannot do the volume terms - chirality and planarity are not pairwise -
 * so this runs FIRST and the gradient pass polishes, which is the division of
 * labour each is good at.
 *
 * @returns {number} the largest violation still outstanding
 */
export function projectOntoBounds(point, bounds, rounds = 200) {
  const { lower, upper, n } = bounds;
  let worst = 0;
  for (let round = 0; round < rounds; round += 1) {
    worst = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = i + 1; j < n; j += 1) {
        const dx = point[i * 3] - point[j * 3];
        const dy = point[i * 3 + 1] - point[j * 3 + 1];
        const dz = point[i * 3 + 2] - point[j * 3 + 2];
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const low = lower[i * n + j];
        const high = upper[i * n + j];
        let target = distance;
        if (distance < low) target = low;
        else if (distance > high) target = high;
        else continue;
        const violation = Math.abs(target - distance);
        if (violation > worst) worst = violation;
        // 🔴 A GUARD ON THE SEPARATION, BECAUSE TWO ATOMS CAN LAND ON TOP OF
        // EACH OTHER. The correction is along the line between them, and when
        // that line has no length there is no direction to move along - the
        // scale would be a division by zero and the coordinates would become
        // NaN, which spreads through every later sweep silently. A nudge along
        // x is as good as any other direction for a pair that is coincident.
        if (distance < 1e-9) {
          point[i * 3] += 0.01;
          continue;
        }
        const scale = (0.5 * (target - distance)) / distance;
        point[i * 3] += scale * dx;
        point[i * 3 + 1] += scale * dy;
        point[i * 3 + 2] += scale * dz;
        point[j * 3] -= scale * dx;
        point[j * 3 + 1] -= scale * dy;
        point[j * 3 + 2] -= scale * dz;
      }
    }
    if (worst < 1e-6) break;
  }
  return worst;
}

/**
 * Push the coordinates until they satisfy the bounds, and the chiral centres
 * have the handedness the SMILES asked for.
 *
 * The error is the usual distance-geometry one: a pair costs nothing while it
 * sits inside its bounds and grows quadratically outside them, normalised by
 * the bound so a long distance and a short one are weighted alike. Steepest
 * descent with a backtracking step - the problem is small and smooth enough
 * that nothing cleverer earns its complexity.
 *
 * @returns {{coordinates: Float64Array, error: number, steps: number}}
 */
export function refineCoordinates(coordinates, bounds, chiral, options = {}) {
  const { lower, upper, n } = bounds;
  const maxSteps = options.steps ?? 400;
  const planar = options.planar ?? [];
  const linear = options.linear ?? [];
  const point = Float64Array.from(coordinates);
  // 🔴 THE PROJECTION FIRST, WHICH IS MOST OF THE ANSWER. See the note on
  // `projectOntoBounds`: the gradient pass reached a satisfied set of bounds
  // for ATP in none of 24 random starts and for a CF3 group in 3.
  projectOntoBounds(point, bounds, options.projections ?? 1000);
  const gradient = new Float64Array(3 * n);
  let step = 0.05;
  let error = boundsError(point, lower, upper, n, chiral, gradient, planar, linear);

  let taken = 0;
  // 🔴 A STALLED LINE SEARCH IS NOT A FINISHED ONE, AND TREATING IT AS ONE WAS
  // WORTH MOST OF THE ERROR. Steepest descent on this objective is stiff - a
  // bond is held to 0.01 A and a torsion to 0.2 - so the step collapses long
  // before the minimum: alanine stopped after 44 steps at an error of 0.264
  // with EVERY bound violated and its bonds 8% short, which reads exactly like
  // a bad embedding rather than an optimiser giving up. Restarting the step
  // size a few times is what gets it the rest of the way, and it is cheap
  // because a stalled search costs one gradient evaluation to discover.
  let restarts = 0;
  for (; taken < maxSteps; taken += 1) {
    const size = Math.hypot(...gradient);
    if (size < 1e-9) break;
    const trial = new Float64Array(3 * n);
    let improved = false;
    for (let attempt = 0; attempt < 20; attempt += 1) {
      for (let i = 0; i < 3 * n; i += 1) {
        trial[i] = point[i] - (step / size) * gradient[i];
      }
      const next = boundsError(trial, lower, upper, n, chiral, null, planar, linear);
      if (next < error) {
        point.set(trial);
        error = next;
        step *= 1.3;                       // it worked; try further next time
        improved = true;
        break;
      }
      step *= 0.4;                         // overshot
    }
    if (!improved) {
      if (restarts >= 4) break;
      restarts += 1;
      step = 0.05;
      continue;
    }
    boundsError(point, lower, upper, n, chiral, gradient, planar, linear);
    if (error < 1e-8) break;
  }
  return { coordinates: point, error, steps: taken };
}

/**
 * The bounds error, and its gradient when one is asked for.
 *
 * 🔴 THE CHIRAL TERM IS A SIGNED VOLUME AND IT IS NOT OPTIONAL. Distance
 * geometry cannot tell an enantiomer from its mirror: every pairwise distance
 * is identical in both, so the embedding returns whichever one the random draw
 * happened to land near, and half of all runs would fold the wrong isomer of a
 * chiral drug. The volume of the tetrahedron on a centre's four neighbours
 * changes sign between them, which is the one quantity that can tell them
 * apart.
 */
function boundsError(point, lower, upper, n, chiral, gradient, planar, linear) {
  if (gradient !== null) gradient.fill(0);
  let error = 0;

  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      const dx = point[i * 3] - point[j * 3];
      const dy = point[i * 3 + 1] - point[j * 3 + 1];
      const dz = point[i * 3 + 2] - point[j * 3 + 2];
      const squared = dx * dx + dy * dy + dz * dz;
      const high = upper[i * n + j];
      const low = lower[i * n + j];
      let scale = 0;
      if (squared > high * high) {
        const over = squared / (high * high) - 1;
        error += over * over;
        scale = (4 * over) / (high * high);
      } else if (squared < low * low) {
        // 🔴 THE LOWER TERM IS WRITTEN OVER `2 l^2 / (l^2 + d^2)` RATHER THAN
        // AS THE MIRROR OF THE UPPER ONE. The naive form's gradient vanishes
        // as two atoms approach each other, so a pair that has collapsed
        // completely feels no force pushing it apart and stays collapsed.
        const ratio = (2 * low * low) / (low * low + squared);
        const over = ratio - 1;
        error += over * over;
        // d(over^2)/d(d^2) is -(over * ratio^2) / l^2, and the chain rule to a
        // coordinate brings a factor of two. The first version carried four
        // and so pushed twice as hard on a short contact as on a long one,
        // which biases the descent direction rather than just its length.
        scale = (-2 * over * ratio * ratio) / (low * low);
      }
      if (gradient !== null && scale !== 0) {
        gradient[i * 3] += scale * dx;
        gradient[i * 3 + 1] += scale * dy;
        gradient[i * 3 + 2] += scale * dz;
        gradient[j * 3] -= scale * dx;
        gradient[j * 3 + 1] -= scale * dy;
        gradient[j * 3 + 2] -= scale * dz;
      }
    }
  }

  // Flat means a signed volume of zero, and unlike the chiral term this one
  // wants the MAGNITUDE driven down rather than the sign driven one way.
  for (const group of planar ?? []) {
    const volume = signedVolume(point, group[0], group[1], group[2], group[3]);
    error += volume * volume * PLANARITY_WEIGHT;
    if (gradient !== null) {
      addVolumeGradient(point, group[0], group[1], group[2], group[3],
                        2 * volume * PLANARITY_WEIGHT, gradient);
    }
  }

  // Straight means the two bond vectors are anti-parallel, so their cross
  // product vanishes. Same shape as the planarity term, one dimension down.
  for (const triple of linear ?? []) {
    const [a, centre, b] = triple;
    const u = [0, 1, 2].map((axis) => point[a * 3 + axis] - point[centre * 3 + axis]);
    const v = [0, 1, 2].map((axis) => point[b * 3 + axis] - point[centre * 3 + axis]);
    const w = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
               u[0] * v[1] - u[1] * v[0]];
    error += (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) * LINEARITY_WEIGHT;
    if (gradient !== null) {
      const cross = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2],
                               p[0] * q[1] - p[1] * q[0]];
      const du = cross(v, w);
      const dv = cross(w, u);
      for (let axis = 0; axis < 3; axis += 1) {
        const pushU = 2 * LINEARITY_WEIGHT * du[axis];
        const pushV = 2 * LINEARITY_WEIGHT * dv[axis];
        gradient[a * 3 + axis] += pushU;
        gradient[b * 3 + axis] += pushV;
        gradient[centre * 3 + axis] -= pushU + pushV;
      }
    }
  }

  for (const centre of chiral ?? []) {
    const [a, b, c, d] = centre.neighbours;
    const volume = signedVolume(point, a, b, c, d);
    const wanted = centre.sign;
    // Only a volume of the wrong sign, or one too flat to be sure of, costs
    // anything: the magnitude is set by the bond lengths and is not this
    // term's business.
    const target = 0.4;
    if (volume * wanted >= target) continue;
    const short = target - volume * wanted;
    error += short * short * 4;
    if (gradient !== null) {
      const push = -8 * short * wanted;
      addVolumeGradient(point, a, b, c, d, push, gradient);
    }
  }
  return error;
}

/** Six times the tetrahedron volume on four atoms, signed. */
export function signedVolume(point, a, b, c, d) {
  const at = (index, axis) => point[index * 3 + axis];
  const ax = at(a, 0) - at(d, 0), ay = at(a, 1) - at(d, 1), az = at(a, 2) - at(d, 2);
  const bx = at(b, 0) - at(d, 0), by = at(b, 1) - at(d, 1), bz = at(b, 2) - at(d, 2);
  const cx = at(c, 0) - at(d, 0), cy = at(c, 1) - at(d, 1), cz = at(c, 2) - at(d, 2);
  return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
}

function addVolumeGradient(point, a, b, c, d, push, gradient) {
  const at = (index, axis) => point[index * 3 + axis];
  const u = [at(a, 0) - at(d, 0), at(a, 1) - at(d, 1), at(a, 2) - at(d, 2)];
  const v = [at(b, 0) - at(d, 0), at(b, 1) - at(d, 1), at(b, 2) - at(d, 2)];
  const w = [at(c, 0) - at(d, 0), at(c, 1) - at(d, 1), at(c, 2) - at(d, 2)];
  const cross = (p, q) => [p[1] * q[2] - p[2] * q[1],
                           p[2] * q[0] - p[0] * q[2],
                           p[0] * q[1] - p[1] * q[0]];
  const da = cross(v, w);
  const db = cross(w, u);
  const dc = cross(u, v);
  for (let axis = 0; axis < 3; axis += 1) {
    gradient[a * 3 + axis] += push * da[axis];
    gradient[b * 3 + axis] += push * db[axis];
    gradient[c * 3 + axis] += push * dc[axis];
    gradient[d * 3 + axis] -= push * (da[axis] + db[axis] + dc[axis]);
  }
}
