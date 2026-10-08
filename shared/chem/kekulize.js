/**
 * Turning a SMILES aromatic ring into alternating single and double bonds.
 *
 * 🔴 THIS RUNS BEFORE THE IMPLICIT HYDROGENS AND THAT ORDER IS THE WHOLE
 * POINT. Counting hydrogens off aromatic bonds worth 1.5 gives caffeine
 * C8H13N4O2 against RDKit's C8H10N4O2: each of its three N-methyl nitrogens
 * sums to 1.5 + 1.5 + 1 = 4, which is over nitrogen's valence of 3, so the
 * rule promotes it to 5 and hands it a hydrogen it does not have. Kekulised,
 * that same nitrogen has two SINGLE ring bonds, sums to 3, and takes none.
 * ATP was off by one hydrogen for the same reason. So aromaticity is resolved
 * into integers first and the valence model only ever sees integers.
 *
 * 🔴 AND A BOND ORDER IS WHAT A BOND LENGTH COMES FROM. `embed.js` places a
 * C-N at 1.47 A single, 1.35 A partial and 1.28 A double; "1.5" is not a row
 * in that table and a ring left aromatic would have no geometry at all.
 * Aromatic rings keep an `aromatic` flag alongside the integer order, because
 * the LENGTH wanted inside a benzene is neither 1.39 nor 1.54 but the
 * delocalised 1.39 - the flag is what says so.
 *
 * The algorithm is the standard one: decide which aromatic atoms still need a
 * double bond, then find a perfect matching over the aromatic bonds joining
 * them. A ring system that has no perfect matching is not kekulisable and is
 * refused by name rather than folded with a wrong bond order.
 */

/**
 * The valence each element is aiming at inside an aromatic ring, before its
 * charge is applied. Only the elements that can BE aromatic are here.
 */
const AROMATIC_VALENCE = { B: 3, C: 4, N: 3, O: 2, P: 3, S: 2, SE: 2, AS: 3 };

/**
 * Resolve every order-1.5 bond into 1 or 2, in place.
 *
 * @param {{atoms: object[], bonds: object[]}} graph from `parseSmiles`, before
 *   its hydrogens are filled in
 */
export function kekulize(graph) {
  const { atoms, bonds } = graph;
  const aromaticBonds = [];
  bonds.forEach((bond, index) => { if (bond.order === 1.5) aromaticBonds.push(index); });
  if (aromaticBonds.length === 0) return;

  // Every bond that is aromatic is also recorded as such, because the LENGTH a
  // delocalised ring wants is not the length either integer order wants.
  for (const index of aromaticBonds) bonds[index].aromatic = true;

  const inAromatic = new Set();
  for (const index of aromaticBonds) {
    inAromatic.add(bonds[index].from);
    inAromatic.add(bonds[index].to);
  }

  // 🔴 WHICH ATOMS NEED A DOUBLE BOND IS THE ONLY CHEMISTRY HERE. An aromatic
  // atom whose sigma bonds, explicit hydrogens and EXOCYCLIC multiple bonds
  // already fill its valence contributes a lone pair instead and must NOT be
  // matched: that is pyrrole's [nH], furan's o, an N-methyl imidazole
  // nitrogen, and - the one that is easy to miss - an aromatic carbon
  // carrying an exocyclic =O, which is how caffeine's two carbonyls sit in an
  // aromatic ring. Matching one of those produces a five-valent carbon.
  const needsDouble = [];
  for (const atom of inAromatic) {
    const info = atoms[atom];
    const valence = AROMATIC_VALENCE[info.symbol];
    if (valence === undefined) continue;
    let used = 0;
    let aromaticNeighbours = 0;
    for (const bond of bonds) {
      const other = bond.from === atom ? bond.to : bond.to === atom ? bond.from : -1;
      if (other < 0) continue;
      if (bond.order === 1.5) { used += 1; aromaticNeighbours += 1; }
      else used += bond.order;
    }
    // A bracketed atom said its hydrogens; an unbracketed one has not been
    // given any yet and is assumed to want one only if it has room.
    const hydrogens = info.bracket ? (info.hydrogens ?? 0) : 0;
    used += hydrogens;
    const target = valence + (info.symbol === "B" ? -info.charge : info.charge);
    // 🔴 AN UNBRACKETED AROMATIC CARBON IS THE DEFAULT AND WANTS A DOUBLE
    // BOND. It is written `c`, it has no stated hydrogen count, and the
    // hydrogen it will get is decided AFTER this - so its own hydrogen must
    // not be counted here or every benzene carbon would look satisfied.
    if (used < target && aromaticNeighbours > 0) needsDouble.push(atom);
  }

  const matched = perfectMatching(needsDouble, aromaticBonds, bonds);
  if (matched === null) {
    const symbols = needsDouble.map((atom) => `${atoms[atom].symbol}${atom}`).join(", ");
    throw new Error("this aromatic ring system cannot be kekulised: no way to give "
      + `each of ${symbols} exactly one double bond. An aromatic atom written `
      + "lower case must be able to take one, or be written with its hydrogen "
      + "in brackets (`[nH]`)");
  }

  for (const index of aromaticBonds) bonds[index].order = matched.has(index) ? 2 : 1;
}

/**
 * Give each atom in `wanted` exactly one bond from `candidates`, or null.
 *
 * A backtracking maximum matching. Blossom would be asymptotically better and
 * is not worth it here: a ligand's aromatic system is tens of atoms, and the
 * degree-ordered heuristic below settles every molecule in this repository's
 * corpus without backtracking at all.
 */
function perfectMatching(wanted, candidates, bonds) {
  if (wanted.length === 0) return new Set();
  if (wanted.length % 2 === 1) return null;   // cannot pair an odd number

  const need = new Set(wanted);
  /** For each wanted atom, the candidate bonds joining it to another wanted one. */
  const options = new Map(wanted.map((atom) => [atom, []]));
  for (const index of candidates) {
    const { from, to } = bonds[index];
    if (need.has(from) && need.has(to)) {
      options.get(from).push(index);
      options.get(to).push(index);
    }
  }

  const taken = new Set();
  const used = new Set();
  const search = () => {
    // Always extend the most constrained atom first, which is what keeps the
    // backtracking from ever starting on an ordinary fused ring system.
    let next = -1;
    let fewest = Infinity;
    for (const atom of need) {
      if (used.has(atom)) continue;
      const open = options.get(atom).filter((index) =>
        !used.has(bonds[index].from) && !used.has(bonds[index].to));
      if (open.length < fewest) { fewest = open.length; next = atom; }
    }
    if (next < 0) return true;                 // everything is paired
    if (fewest === 0) return false;            // this one cannot be
    for (const index of options.get(next)) {
      const { from, to } = bonds[index];
      if (used.has(from) || used.has(to)) continue;
      used.add(from); used.add(to); taken.add(index);
      if (search()) return true;
      used.delete(from); used.delete(to); taken.delete(index);
    }
    return false;
  };
  return search() ? taken : null;
}

/**
 * Check every atom's bonds and hydrogens add up to a valence it can have.
 *
 * 🔴 THIS IS THE ASSERTION A KEKULÉ STRUCTURE ACTUALLY SUPPORTS, and writing
 * the gate any other way was the first mistake here. Benzene has TWO Kekulé
 * structures - 1=2, 3=4, 5=6 and 2=3, 4=5, 6=1 - and they are the same
 * molecule; which one a program picks is an artefact of the order it searched
 * in. Comparing this port's chosen orders bond by bond against RDKit's failed
 * six of fifty-one molecules and every single difference was that arbitrary
 * choice, with the formula, the hydrogen counts and the rings all agreeing.
 *
 * So the invariant is not "the same double bonds as RDKit". It is "every atom
 * ends up with a valence it can have", which a wrong kekulisation breaks and a
 * different-but-valid one does not.
 *
 * @returns {string[]} one message per atom that does not add up; empty is good
 */
export function valenceProblems(graph) {
  const ALLOWED = {
    B: [3], C: [4], N: [3, 4, 5], O: [2], F: [1], SI: [4], P: [3, 5],
    S: [2, 4, 6], CL: [1, 3, 5, 7], SE: [2, 4, 6], BR: [1, 3, 5, 7],
    I: [1, 3, 5, 7], AS: [3, 5],
  };
  const totals = graph.atoms.map(() => 0);
  for (const bond of graph.bonds) {
    totals[bond.from] += bond.order;
    totals[bond.to] += bond.order;
  }
  const problems = [];
  graph.atoms.forEach((atom, index) => {
    const allowed = ALLOWED[atom.symbol];
    if (allowed === undefined) return;               // a metal; no valence rule
    const used = totals[index] + (atom.hydrogens ?? 0);
    // A charge shifts what is allowed: [NH4+] is four, [O-] is one.
    const shift = atom.symbol === "B" ? -atom.charge : atom.charge;
    if (!allowed.some((valence) => valence + shift === used)) {
      problems.push(`atom ${index} ${atom.symbol}`
        + `${atom.charge === 0 ? "" : atom.charge > 0 ? `+${atom.charge}` : atom.charge}`
        + ` has valence ${used}, and can have `
        + `${allowed.map((valence) => valence + shift).join(" or ")}`);
    }
  });
  return problems;
}

/**
 * Average the bond orders across a delocalised charged group, in place.
 *
 * 🔴 A CARBOXYLATE HAS TWO IDENTICAL C-O BONDS AND SMILES CANNOT SAY SO.
 * `CC(=O)[O-]` writes one double and one single, which is a resonance form and
 * not a description: the two oxygens are indistinguishable and both bonds are
 * 1.25 A. Placed as written they come out 1.16 and 1.34, a tenth of an
 * angstrom apart in a group where the real difference is zero - and an
 * acetate with one long and one short arm is exactly what `bond-geometry.js`
 * reports as a broken ligand.
 *
 * Measured against RDKit's MMFF conformers over the corpus, this is the single
 * largest source of error in the bond lengths: the sulfonate S-O was 0.24 A
 * out before it, the worst of any bond kind, and the nitro N-O pair was
 * +0.09 / -0.11 in opposite directions.
 *
 * The rule: a central atom carrying several TERMINAL oxygens or sulfurs, at
 * least one of them double-bonded and at least one single, delocalises over
 * all of them. Terminal is the crucial qualifier - an ester's -O-C is not part
 * of the group and must keep its single bond, which is what distinguishes
 * `CC(=O)OC` (1.20 and 1.34, genuinely different) from `CC(=O)[O-]`.
 */
export function delocalizeCharges(graph) {
  const { atoms, bonds } = graph;
  const degree = atoms.map(() => 0);
  for (const bond of bonds) { degree[bond.from] += 1; degree[bond.to] += 1; }

  for (let centre = 0; centre < atoms.length; centre += 1) {
    if (!["C", "N", "S", "P"].includes(atoms[centre].symbol)) continue;
    const arms = [];
    bonds.forEach((bond, index) => {
      const other = bond.from === centre ? bond.to
        : bond.to === centre ? bond.from : -1;
      if (other < 0) return;
      if (degree[other] !== 1) return;                     // not terminal
      if (!["O", "S"].includes(atoms[other].symbol)) return;
      if (atoms[other].hydrogens > 0) return;              // an -OH is not in it
      arms.push({ index, order: bond.order });
    });
    if (arms.length < 2) continue;
    const hasDouble = arms.some((arm) => arm.order >= 2);
    const hasSingle = arms.some((arm) => arm.order === 1);
    if (!hasDouble || !hasSingle) continue;
    const mean = arms.reduce((total, arm) => total + arm.order, 0) / arms.length;
    for (const arm of arms) {
      // 🔴 IT SETS A SECOND FIELD AND LEAVES `order` ALONE. The integer order
      // is the molecule as written and is what the valence check adds up and
      // what the RDKit gate compares; a fractional one there would make both
      // meaningless. `geometryOrder` is consulted by the bond-length table and
      // by nothing else, which is the only place the delocalisation is true.
      bonds[arm.index].geometryOrder = mean;
      bonds[arm.index].resonance = true;
    }
  }
}
