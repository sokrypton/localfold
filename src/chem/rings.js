/**
 * Ring perception: the smallest set of smallest rings, and what needs it.
 *
 * 🔴 RINGS ARE NOT DECORATION HERE, THEY ARE GEOMETRY. `embed.js` builds a
 * distance bound for every pair of atoms, and for a pair inside a ring the
 * bound is not "somewhere between a bonded length and infinity" - a
 * cyclohexane's 1,4 distance is pinned within a few hundredths, and a benzene
 * is FLAT, which fixes every distance in it exactly. Without ring perception
 * the embedding has nothing to stop a six-ring coming out as a knot, and the
 * bond-geometry gate downstream reads that as a torn ligand.
 *
 * The count is fixed before the search starts: a connected graph has
 * `bonds - atoms + 1` independent cycles, so the answer has exactly that many
 * rings per fragment (the cyclomatic number, or circuit rank). Finding them is
 * then a matter of collecting short cycles and keeping the ones that are
 * linearly independent over GF(2) on the bond space, which is Horton's
 * algorithm with the usual smallest-first ordering.
 *
 * 🔴 AND "SYMMETRIC SSSR" IS A DIFFERENT QUESTION THAT LOOKS LIKE THE SAME
 * ONE. Adamantane has a circuit rank of THREE and four equally small rings,
 * and no choice of three is more correct than another - RDKit's
 * `GetSymmSSSR` returns all four for exactly that reason. This returns the
 * symmetric set too: for geometry, a ring constraint left out because an
 * arbitrary tie-break dropped it is a ring left floppy.
 */

import { adjacency } from "./smiles.js";

/**
 * The smallest set of smallest rings, each as an array of atom indices in
 * cycle order.
 *
 * @param {{atoms: object[], bonds: object[]}} graph
 * @returns {number[][]}
 */
export function smallestRings(graph) {
  const lists = adjacency(graph);
  const candidates = shortestCycles(graph, lists);

  // Independence is tested over the BOND space: a ring is a set of bonds, and
  // a cycle basis is what the circuit rank counts. Testing over atoms instead
  // calls two different rings through the same atoms dependent.
  candidates.sort((a, b) => a.atoms.length - b.atoms.length);

  const chosen = [];
  /** Reduced bond vectors, each tagged with the ring size it came from. */
  const basis = [];
  const rank = circuitRank(graph);
  let independent = 0;

  /** Reduce `vector` against basis rows from rings no larger than `limit`. */
  const reduce = (start, limit) => {
    let vector = start;
    for (const row of basis) {
      if (row.size > limit) continue;
      const next = vector ^ row.vector;
      if (next < vector) vector = next;
    }
    return vector;
  };

  for (const candidate of candidates) {
    const size = candidate.atoms.length;
    const full = reduce(candidate.vector, Infinity);
    if (full !== 0n) {
      if (independent >= rank) continue;
      basis.push({ vector: full, size });
      basis.sort((a, b) => (b.vector > a.vector ? 1 : b.vector < a.vector ? -1 : 0));
      chosen.push(candidate.atoms);
      independent += 1;
      continue;
    }
    // 🔴 A DEPENDENT RING IS KEPT ONLY WHEN NO SMALLER RING IS NEEDED TO MAKE
    // IT, WHICH IS NOT THE SAME AS "SAME SIZE AS ONE ALREADY IN". That looser
    // rule is what makes the set symmetric for a cage - adamantane's fourth
    // six-ring is the sum of the other three over GF(2) and is just as real a
    // ring, and dropping it leaves a face unconstrained when the embedder
    // reads these as geometry. But it also kept a ring that is not a ring in
    // any useful sense: taxol's eight-membered candidate is exactly its
    // six-ring PLUS its four-ring, a lap around two fused faces, and this
    // port returned 8 rings where RDKit returns 7.
    //
    // The test that separates them is whether the reduction needs anything
    // SMALLER. Adamantane's fourth ring reduces using only six-rings; taxol's
    // eight reduces to zero the moment its four-ring is allowed in. So the
    // candidate is reduced twice - once against smaller rings alone, once
    // against everything - and only a ring that survives the first is a
    // genuine alternative rather than a composite.
    if (reduce(candidate.vector, size - 1) === 0n) continue;
    if (chosen.some((ring) => sameRing(ring, candidate.atoms))) continue;
    chosen.push(candidate.atoms);
  }
  return chosen;
}

/** `bonds - atoms + fragments`, the number of independent cycles. */
export function circuitRank(graph) {
  const fragments = new Set(graph.components
    ?? new Array(graph.atoms.length).fill(0)).size;
  return graph.bonds.length - graph.atoms.length + fragments;
}

const sameRing = (a, b) =>
  a.length === b.length && new Set(a).size === new Set([...a, ...b]).size;

/**
 * Horton's candidate pool: for every atom and every bond, the cycle made of
 * the two shortest paths from that atom to the bond's ends, plus the bond.
 *
 * 🔴 ONE SHORTEST CYCLE PER BOND IS NOT ENOUGH OF A POOL, AND CUBANE IS THE
 * PROOF. Every bond of a cubane lies on TWO of its six square faces, and a
 * breadth-first search returns whichever it reaches first - so over twelve
 * bonds the pool held only four distinct faces where the circuit rank is five,
 * and the basis came out one ring short of what RDKit's `GetSymmSSSR`
 * returns. The pool has to contain every ring that might be in the answer,
 * and per-bond search does not promise that.
 *
 * Horton's does: every ring in a minimum cycle basis is, for some atom on it,
 * exactly this construction. It is O(V*E) candidates against O(E), which for
 * a ligand is a few thousand short walks and costs nothing measurable.
 *
 * 🔴 AND THE TWO PATHS MUST MEET ONLY AT THE ROOT. Glued together when they
 * share a later atom, the result is a figure of eight rather than a cycle -
 * it has the right bond count and is not a ring, and it would enter the basis
 * as one.
 */
function shortestCycles(graph, lists) {
  const found = new Map();
  const n = graph.atoms.length;
  for (let root = 0; root < n; root += 1) {
    const { previous, depth } = breadthFirst(lists, root);
    graph.bonds.forEach((bond, index) => {
      const { from, to } = bond;
      if (depth[from] === undefined || depth[to] === undefined) return;
      const left = pathTo(previous, from);
      const right = pathTo(previous, to);
      // Internally disjoint: the root is the only atom they share.
      const seen = new Set(left);
      let shared = 0;
      for (const atom of right) if (seen.has(atom)) shared += 1;
      if (shared !== 1) return;
      // left runs root..from, right runs root..to; the ring is
      // from..root..to plus the bond (to, from).
      const atoms = [...left.slice().reverse(), ...right.slice(1)];
      if (atoms.length < 3) return;
      const key = [...atoms].sort((a, b) => a - b).join(",");
      if (found.has(key)) return;
      found.set(key, { atoms, vector: bondVector(atoms, graph) });
      void index;
    });
  }
  return [...found.values()];
}

/** Breadth-first from `root`, keeping each atom's parent and depth. */
function breadthFirst(lists, root) {
  const previous = new Map([[root, -1]]);
  const depth = [];
  depth[root] = 0;
  const queue = [root];
  for (let head = 0; head < queue.length; head += 1) {
    const atom = queue[head];
    for (const step of lists[atom]) {
      if (previous.has(step.atom)) continue;
      previous.set(step.atom, atom);
      depth[step.atom] = depth[atom] + 1;
      queue.push(step.atom);
    }
  }
  return { previous, depth };
}

/** The path from the search root out to `atom`, root first. */
function pathTo(previous, atom) {
  const path = [];
  for (let at = atom; at !== -1 && at !== undefined; at = previous.get(at)) path.push(at);
  return path.reverse();
}

/** The shortest path from `start` to `goal` that does not use bond `without`. */
function shortestPath(lists, start, goal, without) {
  const previous = new Map([[start, -1]]);
  const queue = [start];
  for (let head = 0; head < queue.length; head += 1) {
    const atom = queue[head];
    if (atom === goal) break;
    for (const step of lists[atom]) {
      if (step.bond === without || previous.has(step.atom)) continue;
      previous.set(step.atom, atom);
      queue.push(step.atom);
    }
  }
  if (!previous.has(goal)) return null;
  const path = [];
  for (let atom = goal; atom !== -1; atom = previous.get(atom)) path.push(atom);
  return path;
}

/** A ring's bonds as a bit per bond, so independence is an XOR. */
function bondVector(ring, graph) {
  const inRing = new Set();
  for (let index = 0; index < ring.length; index += 1) {
    const a = ring[index];
    const b = ring[(index + 1) % ring.length];
    inRing.add(a < b ? `${a}-${b}` : `${b}-${a}`);
  }
  let vector = 0n;
  graph.bonds.forEach((bond, index) => {
    const key = bond.from < bond.to
      ? `${bond.from}-${bond.to}` : `${bond.to}-${bond.from}`;
    if (inRing.has(key)) vector |= 1n << BigInt(index);
  });
  return vector;
}

/** Which ring sizes each atom belongs to, for the geometry model. */
export function ringMembership(graph, rings) {
  const sizes = Array.from({ length: graph.atoms.length }, () => []);
  for (const ring of rings) for (const atom of ring) sizes[atom].push(ring.length);
  return sizes;
}

/** True for each bond that lies in at least one ring of `rings`. */
export function bondsInRings(graph, rings) {
  const inRing = new Set();
  for (const ring of rings) {
    for (let index = 0; index < ring.length; index += 1) {
      const a = ring[index];
      const b = ring[(index + 1) % ring.length];
      inRing.add(a < b ? `${a}-${b}` : `${b}-${a}`);
    }
  }
  return graph.bonds.map((bond) => inRing.has(
    bond.from < bond.to ? `${bond.from}-${bond.to}` : `${bond.to}-${bond.from}`));
}
