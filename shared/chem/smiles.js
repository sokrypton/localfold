/**
 * SMILES, parsed into a molecular graph.
 *
 * 🔴 THIS IS THE FRONT DOOR FOR A LIGAND NOBODY HAS A CCD CODE FOR, and the
 * whole point of the exercise: `parseCcdComponent` returns
 * `{code, atoms: [{name, element, charge, x, y, z, leaving}], bonds: [{from,
 * to, order}]}` and every stage after it - the atomised tokens, the bond
 * matrix, the chirality centres, `bond-geometry.js`'s scoring - reads that and
 * nothing else. So a SMILES ligand does not need a second featuriser: it needs
 * to arrive in exactly that shape. This file does the first half, the graph;
 * `embed.js` does the coordinates and `component.js` assembles the two.
 *
 * 🔴 AND HYDROGENS ARE COUNTED HERE AND DROPPED LATER. AF3 tokenises a ligand
 * one token per HEAVY atom, so the component that comes out carries no
 * hydrogens at all - but the count per heavy atom decides the geometry
 * (a nitrogen with one H is planar in an amide and pyramidal in an amine), so
 * it is parsed, derived where implicit, and kept on the atom.
 *
 * Follows the OpenSMILES specification. What is deliberately NOT here:
 * reaction SMILES (`>>`), wildcard `*` in ring-closure position, and the
 * extended stereo classes (`@TH`, `@AL`, `@SP`, `@TB`, `@OH`) beyond plain
 * tetrahedral - a ligand needing one of those is refused by name rather than
 * read wrongly, which is this repository's rule about a convention it does not
 * implement.
 */

import { ELEMENT_SYMBOLS } from "../af3/featurise/ccd-component.js";
import { delocalizeCharges, kekulize } from "./kekulize.js";

/**
 * The organic subset: these may appear outside brackets and take their
 * hydrogen count from the valence model rather than from the string.
 */
const ORGANIC = new Set(["B", "C", "N", "O", "P", "S", "F", "CL", "BR", "I"]);

/**
 * Lowercase aromatic atoms, which are the organic subset that can be aromatic
 * plus the ones only legal inside brackets (`se`, `as`).
 */
const AROMATIC_ORGANIC = new Set(["b", "c", "n", "o", "p", "s"]);

/**
 * 🔴 THE VALENCES THAT DECIDE AN IMPLICIT HYDROGEN COUNT, and the list is the
 * specification's rather than chemistry's in general: OpenSMILES fixes exactly
 * these, and an atom outside the organic subset has NO implicit hydrogens at
 * all - it must spell them in its brackets. That asymmetry is the whole reason
 * `[Fe]` and `C` behave differently and is not a special case to smooth over.
 *
 * Several valences means the lowest one that is not exceeded: sulfur is 2 in a
 * thioether, 4 in a sulfoxide and 6 in a sulfone, and the bond count is what
 * says which.
 */
const IMPLICIT_VALENCES = {
  B: [3], C: [4], N: [3, 5], O: [2], P: [3, 5], S: [2, 4, 6],
  F: [1], CL: [1], BR: [1], I: [1],
};

/** A bond symbol's order. `:` is aromatic and is resolved by kekulisation. */
const BOND_SYMBOLS = { "-": 1, "=": 2, "#": 3, $: 4, ":": 1.5, "/": 1, "\\": 1 };

/**
 * One atom of the parsed graph.
 *
 * @typedef {object} SmilesAtom
 * @property {string} symbol element symbol, upper case
 * @property {number} element one-based index into ELEMENT_SYMBOLS
 * @property {boolean} aromatic written lower case, or `[se]`-style
 * @property {number} charge formal charge
 * @property {number} isotope 0 when unspecified
 * @property {number|null} hydrogens explicit count from brackets; null means
 *   "derive it", which only the organic subset may say
 * @property {"@"|"@@"|null} chirality as written, before the neighbour order
 *   is known - see `tetrahedralParity`
 * @property {number} mapClass the `:n` atom class, 0 when unspecified
 * @property {boolean} bracket whether it was written in brackets
 */

/**
 * One bond of the parsed graph.
 *
 * @typedef {object} SmilesBond
 * @property {number} from
 * @property {number} to
 * @property {number} order 1, 2, 3, 4, or 1.5 for an unkekulised aromatic bond
 * @property {"/"|"\\"|null} direction the cis/trans marker as written
 * @property {boolean} ring whether it closed a ring rather than continuing
 *   the chain - kept because a ring-closure bond's DIRECTION is written from
 *   the other end and a stereo perception has to know
 */

/**
 * Read one SMILES string.
 *
 * @param {string} text
 * @returns {{atoms: SmilesAtom[], bonds: SmilesBond[], components: number[]}}
 *   `components` labels each atom with its disconnected fragment, since `.`
 *   is legal and a two-fragment ligand is a real thing (a salt).
 */
export function parseSmiles(text) {
  if (typeof text !== "string") throw new Error("a SMILES must be a string");
  const smiles = text.trim();
  if (smiles === "") throw new Error("an empty SMILES");
  if (smiles.includes(">")) {
    throw new Error("reaction SMILES (`>`) is not a ligand");
  }

  const atoms = [];
  const bonds = [];
  /**
   * 🔴 THE ORDER NEIGHBOURS WERE WRITTEN IN, WHICH IS NOT THE ORDER THE BONDS
   * WERE CREATED IN, and tetrahedral chirality is defined by the first. A ring
   * closure is WRITTEN as a digit on the opening atom and the bond is CREATED
   * later, when the ring closes - `[C@@H]1SC...1` writes its ring neighbour
   * second and creates that bond last. Sorting a centre's neighbours by bond
   * index therefore puts the ring bond at the end and INVERTS the centre: 9 of
   * 29 corpus centres came out as the wrong enantiomer, all of them ring
   * openings, with every bond length and angle still perfect. Each entry is a
   * bond index, or a placeholder object that the ring closure fills in.
   */
  const writtenNeighbours = [];
  /** Where each ring-closure digit was opened: label -> {atom, order, direction, index}. */
  const openRings = new Map();
  /** The branch stack; each entry is the atom a `)` returns to. */
  const branches = [];
  /** The atom the next one bonds back to, or null at the start of a fragment. */
  let previous = null;
  /** A bond symbol seen but not yet used. */
  let pending = null;
  let position = 0;

  const fail = (message) => {
    throw new Error(`${message} at position ${position} of "${smiles}"`);
  };

  /** Record a bond, refusing a duplicate, which SMILES does not permit. */
  const addBond = (from, to, order, direction, ring) => {
    if (from === to) fail("an atom bonded to itself");
    for (const bond of bonds) {
      if ((bond.from === from && bond.to === to)
        || (bond.from === to && bond.to === from)) {
        fail(`atoms ${from} and ${to} are bonded twice`);
      }
    }
    bonds.push({ from, to, order, direction: direction ?? null, ring: ring === true });
  };

  while (position < smiles.length) {
    const character = smiles[position];

    if (character === "(") {
      if (previous === null) fail("a branch before any atom");
      branches.push(previous);
      position += 1;
      continue;
    }
    if (character === ")") {
      if (branches.length === 0) fail("a `)` with no `(`");
      previous = branches.pop();
      position += 1;
      continue;
    }
    if (character === ".") {
      // 🔴 A DOT STARTS A NEW FRAGMENT AND DOES NOT END A BRANCH. `C(.C)C` is
      // legal and the dot only breaks the bond to what came before.
      previous = null;
      pending = null;
      position += 1;
      continue;
    }
    if (character in BOND_SYMBOLS) {
      if (pending !== null) fail("two bond symbols in a row");
      pending = character;
      position += 1;
      continue;
    }
    if (character === "%" || /[0-9]/.test(character)) {
      if (previous === null) fail("a ring closure before any atom");
      let label;
      if (character === "%") {
        const digits = smiles.slice(position + 1, position + 3);
        if (!/^[0-9]{2}$/.test(digits)) fail("`%` needs two digits");
        label = digits;
        position += 3;
      } else {
        label = character;
        position += 1;
      }
      const open = openRings.get(label);
      if (open === undefined) {
        // The slot is reserved HERE, where the digit is written, and filled in
        // when the ring closes. That is the whole fix for the inverted centres.
        const slot = { pendingRing: true };
        writtenNeighbours[previous].push(slot);
        openRings.set(label, { atom: previous, symbol: pending, slot });
        pending = null;
        continue;
      }
      // 🔴 EITHER END MAY CARRY THE BOND SYMBOL AND THEY MUST AGREE. `C=1CCC1`
      // and `C1CCC=1` are the same molecule; `C=1CCC#1` is an error rather
      // than a choice, and reading it as one silently would change the bond.
      openRings.delete(label);
      const here = pending;
      pending = null;
      let symbol = open.symbol ?? here;
      if (open.symbol !== null && open.symbol !== undefined
        && here !== null && open.symbol !== here) {
        // ...unless they are the two directional marks, which are written from
        // opposite ends and so are expected to differ; see `oppositeOf`.
        if (!(isDirection(open.symbol) && isDirection(here))) {
          fail(`ring bond ${label} is ${open.symbol} at one end and ${here} at the other`);
        }
        symbol = open.symbol;
      }
      const bothAromatic = atoms[open.atom].aromatic && atoms[previous].aromatic;
      const order = symbol === null || symbol === undefined
        ? (bothAromatic ? 1.5 : 1) : BOND_SYMBOLS[symbol];
      // A ring-closure direction belongs to the atom that WROTE it, so the
      // opening end's mark is the one stored and the closing end's is flipped.
      const direction = isDirection(open.symbol) ? open.symbol
        : (isDirection(here) ? oppositeOf(here) : null);
      addBond(open.atom, previous, order, direction, true);
      const created = bonds.length - 1;
      const at = writtenNeighbours[open.atom].indexOf(open.slot);
      if (at >= 0) writtenNeighbours[open.atom][at] = created;
      writtenNeighbours[previous].push(created);
      continue;
    }

    // ...otherwise it is an atom.
    const atom = readAtom(smiles, position, fail);
    position = atom.next;
    const index = atoms.length;
    atoms.push(atom.atom);
    writtenNeighbours.push([]);
    if (previous !== null) {
      const bothAromatic = atoms[previous].aromatic && atom.atom.aromatic;
      const order = pending === null
        ? (bothAromatic ? 1.5 : 1) : BOND_SYMBOLS[pending];
      addBond(previous, index, order, isDirection(pending) ? pending : null, false);
      writtenNeighbours[previous].push(bonds.length - 1);
      writtenNeighbours[index].push(bonds.length - 1);
    } else if (pending !== null) {
      fail("a bond symbol at the start of a fragment");
    }
    pending = null;
    previous = index;
  }

  if (branches.length > 0) fail("a `(` with no `)`");
  if (openRings.size > 0) {
    throw new Error(`ring closure${openRings.size === 1 ? "" : "s"} `
      + `${[...openRings.keys()].join(", ")} never closed in "${smiles}"`);
  }
  if (pending !== null) fail("a bond symbol at the end");
  if (atoms.length === 0) throw new Error(`no atoms in "${smiles}"`);

  // 🔴 AN EXPLICIT HYDROGEN IS A HYDROGEN, NOT AN ATOM OF THE COMPONENT.
  // `[2H]C([2H])([2H])O` is methanol with three deuteriums, which RDKit counts
  // as TWO heavy atoms and this port counted as FIVE - because the parser
  // kept every bracket atom, and `[2H]` is written like one. That reaches the
  // featuriser as a ligand with three extra tokens, since AF3 tokenises one
  // token per HEAVY atom and `parseCcdComponent` drops hydrogens before
  // anyone sees them. Merged here, before kekulisation, because the count
  // changes what valence every neighbour has left.
  mergeExplicitHydrogens(atoms, bonds, writtenNeighbours);

  // 🔴 KEKULISE FIRST, THEN COUNT HYDROGENS. See the note at the top of
  // kekulize.js: hydrogens counted off bonds worth 1.5 gave caffeine three it
  // does not have and ATP one, because a sum of 4 over nitrogen's valence of 3
  // promotes it to 5. With integer orders the same atoms sum to 3 and take
  // none. The order is not a preference; it is the difference between right
  // and wrong for every aromatic nitrogen that carries a substituent.
  kekulize({ atoms, bonds });
  fillImplicitHydrogens(atoms, bonds);
  // ...and only then the resonance, which needs the hydrogens to know that an
  // -OH is not part of a carboxylate. It writes `geometryOrder`, never `order`.
  delocalizeCharges({ atoms, bonds });
  return {
    atoms, bonds, writtenNeighbours,
    components: fragmentsOf(atoms.length, bonds),
  };
}

const isDirection = (symbol) => symbol === "/" || symbol === "\\";
const oppositeOf = (symbol) => (symbol === "/" ? "\\" : "/");

/**
 * Read one atom, bracketed or not, starting at `start`.
 *
 * @returns {{atom: SmilesAtom, next: number}}
 */
function readAtom(smiles, start, fail) {
  if (smiles[start] !== "[") {
    // The organic subset, and the two-letter ones must be tried first or
    // `Cl` reads as carbon followed by an unknown `l`.
    for (const symbol of ["Cl", "Br"]) {
      if (smiles.startsWith(symbol, start)) {
        return { atom: makeAtom(symbol.toUpperCase(), false, 0, 0, null, null, 0, false),
                 next: start + 2 };
      }
    }
    const one = smiles[start];
    if (ORGANIC.has(one.toUpperCase()) && one === one.toUpperCase()) {
      return { atom: makeAtom(one.toUpperCase(), false, 0, 0, null, null, 0, false),
               next: start + 1 };
    }
    if (AROMATIC_ORGANIC.has(one)) {
      return { atom: makeAtom(one.toUpperCase(), true, 0, 0, null, null, 0, false),
               next: start + 1 };
    }
    if (one === "*") {
      // 🔴 A WILDCARD IS REFUSED RATHER THAN GUESSED. It means "any atom", and
      // a ligand with an unspecified element has no conformer and no mass; the
      // featuriser downstream would need an element index for it.
      fail("`*` (any atom) has no element and cannot be folded");
    }
    fail(`"${one}" is not an organic-subset atom; bracket it`);
  }

  const close = smiles.indexOf("]", start);
  if (close < 0) fail("a `[` with no `]`");
  const body = smiles.slice(start + 1, close);
  const match = body.match(
    /^(\d*)([A-Za-z][a-z]?)(@{1,2}(?:TH|AL|SP|TB|OH)?\d*)?(H\d*)?((?:[+-]\d*|\++|-+)?)(?::(\d+))?$/);
  if (match === null) fail(`cannot read the bracket atom "[${body}]"`);
  const [, isotope, written, chiralRaw, hydrogenRaw, chargeRaw, mapRaw] = match;

  const aromatic = written[0] === written[0].toLowerCase()
    && /[a-z]/.test(written[0]);
  const symbol = written.toUpperCase();
  if (ELEMENT_SYMBOLS.indexOf(symbol) < 0) {
    fail(`"${written}" is not an element`);
  }

  let chirality = null;
  if (chiralRaw !== undefined) {
    if (/^@{1,2}$/.test(chiralRaw)) {
      chirality = chiralRaw;
    } else {
      // 🔴 REFUSED BY NAME, NOT READ AS PLAIN TETRAHEDRAL. `@TB3` is a
      // trigonal-bipyramidal arrangement; treating it as `@` would place four
      // of five neighbours by a rule that does not apply to them.
      fail(`stereo class "${chiralRaw}" is not supported; only @ and @@ are`);
    }
  }

  let hydrogens = 0;
  if (hydrogenRaw !== undefined) {
    hydrogens = hydrogenRaw === "H" ? 1 : Number.parseInt(hydrogenRaw.slice(1), 10);
  }

  let charge = 0;
  if (chargeRaw !== undefined && chargeRaw !== "") {
    if (/^[+-]\d+$/.test(chargeRaw)) charge = Number.parseInt(chargeRaw, 10);
    else charge = (chargeRaw[0] === "+" ? 1 : -1) * chargeRaw.length;
  }

  return {
    atom: makeAtom(symbol, aromatic, charge,
                   isotope === "" ? 0 : Number.parseInt(isotope, 10),
                   hydrogens, chirality,
                   mapRaw === undefined ? 0 : Number.parseInt(mapRaw, 10), true),
    next: close + 1,
  };
}

function makeAtom(symbol, aromatic, charge, isotope, hydrogens, chirality, mapClass, bracket) {
  return {
    symbol,
    element: ELEMENT_SYMBOLS.indexOf(symbol) + 1,
    aromatic,
    charge,
    isotope,
    hydrogens,
    chirality,
    mapClass,
    bracket,
  };
}

/**
 * Give every organic-subset atom written without brackets its hydrogen count.
 *
 * 🔴 AN AROMATIC ATOM'S VALENCE IS COUNTED WITH ITS AROMATIC BONDS AS 1.5 AND
 * THEN ROUNDED UP, which is the specification's rule and not an approximation:
 * benzene's carbon has two 1.5 bonds summing to 3, needs 4, and takes one
 * hydrogen. Pyridine's nitrogen sums to 3, needs 3, and takes none. Pyrrole's
 * nitrogen also sums to 3 - and takes one, because its valence target is 3 and
 * the ring contributes only 2 sigma bonds; that case is why the sum is rounded
 * UP rather than down.
 */
function fillImplicitHydrogens(atoms, bonds) {
  const order = atoms.map(() => 0);
  const degree = atoms.map(() => 0);
  for (const bond of bonds) {
    order[bond.from] += bond.order;
    order[bond.to] += bond.order;
    degree[bond.from] += 1;
    degree[bond.to] += 1;
  }
  atoms.forEach((atom, index) => {
    if (atom.hydrogens !== null) return;          // bracketed: it said so
    const valences = IMPLICIT_VALENCES[atom.symbol];
    if (valences === undefined) { atom.hydrogens = 0; return; }
    // An aromatic atom in a ring of 1.5-order bonds sums to a half-integer.
    const used = Math.ceil(order[index] - 1e-9);
    // ...and a charge moves the target: [NH4+] is nitrogen with four bonds.
    const shift = atom.symbol === "B" ? -atom.charge : atom.charge;
    const target = valences.find((valence) => valence + shift >= used);
    atom.hydrogens = target === undefined
      ? 0 : Math.max(0, target + shift - used);
  });
}

/**
 * Fold every `[H]` written as its own atom into its neighbour's count.
 *
 * 🔴 EXCEPT WHEN IT HAS NOWHERE TO GO. Molecular hydrogen `[H][H]` is two
 * hydrogens bonded to each other and neither is a substituent of the other; a
 * lone `[H+]` is a proton. Those stay as atoms, because turning them into a
 * count would delete the molecule. Only a hydrogen with exactly one neighbour
 * that is NOT itself a hydrogen is a substituent.
 */
function mergeExplicitHydrogens(atoms, bonds, writtenNeighbours) {
  const neighbours = atoms.map(() => []);
  bonds.forEach((bond, index) => {
    neighbours[bond.from].push({ atom: bond.to, bond: index });
    neighbours[bond.to].push({ atom: bond.from, bond: index });
  });
  const drop = new Set();
  atoms.forEach((atom, index) => {
    if (atom.symbol !== "H") return;
    if (neighbours[index].length !== 1) return;
    const host = neighbours[index][0].atom;
    if (atoms[host].symbol === "H") return;
    // A charged or mapped hydrogen is being singled out deliberately; folding
    // it into a count would throw that away silently.
    if (atom.charge !== 0) return;
    atoms[host].hydrogens = (atoms[host].hydrogens ?? 0) + 1;
    drop.add(index);
  });
  if (drop.size === 0) return;

  // 🔴 AND EVERYTHING THAT INDEXES AN ATOM IS RENUMBERED, which is the half
  // that is easy to forget: the bonds, and the written-neighbour lists that
  // tetrahedral chirality reads. `polymerResidue` has the identical comment
  // about dropping a leaving atom - "dropping an atom without remapping them
  // silently rewires the residue" - and it is the same hazard here.
  const moved = [];
  let next = 0;
  for (let index = 0; index < atoms.length; index += 1) {
    moved.push(drop.has(index) ? -1 : next);
    if (!drop.has(index)) next += 1;
  }
  const keptBonds = [];
  const bondMoved = [];
  bonds.forEach((bond) => {
    if (drop.has(bond.from) || drop.has(bond.to)) { bondMoved.push(-1); return; }
    bondMoved.push(keptBonds.length);
    keptBonds.push({ ...bond, from: moved[bond.from], to: moved[bond.to] });
  });
  const keptWritten = [];
  writtenNeighbours.forEach((list, index) => {
    if (drop.has(index)) return;
    keptWritten.push(list
      .map((entry) => (typeof entry === "number" ? bondMoved[entry] : entry))
      .filter((entry) => entry !== -1));
  });
  const keptAtoms = atoms.filter((_, index) => !drop.has(index));
  atoms.length = 0; atoms.push(...keptAtoms);
  bonds.length = 0; bonds.push(...keptBonds);
  writtenNeighbours.length = 0; writtenNeighbours.push(...keptWritten);
}

/** Label each atom with the index of its disconnected fragment. */
function fragmentsOf(count, bonds) {
  const label = new Array(count).fill(-1);
  const neighbours = Array.from({ length: count }, () => []);
  for (const bond of bonds) {
    neighbours[bond.from].push(bond.to);
    neighbours[bond.to].push(bond.from);
  }
  let next = 0;
  for (let start = 0; start < count; start += 1) {
    if (label[start] >= 0) continue;
    const stack = [start];
    label[start] = next;
    while (stack.length > 0) {
      const atom = stack.pop();
      for (const other of neighbours[atom]) {
        if (label[other] < 0) { label[other] = next; stack.push(other); }
      }
    }
    next += 1;
  }
  return label;
}

/** Neighbour lists, each entry `{atom, order, bond}`, in bond order. */
export function adjacency(graph) {
  const lists = Array.from({ length: graph.atoms.length }, () => []);
  graph.bonds.forEach((bond, index) => {
    lists[bond.from].push({ atom: bond.to, order: bond.order, bond: index });
    lists[bond.to].push({ atom: bond.from, order: bond.order, bond: index });
  });
  return lists;
}

/**
 * The molecular formula, heavy atoms and their hydrogens, in Hill order.
 *
 * Here because it is the cheapest possible check that a parse is right, and
 * `tools/check-smiles-vs-rdkit.mjs` compares exactly this against RDKit's own.
 */
export function molecularFormula(graph) {
  const counts = new Map();
  let hydrogens = 0;
  for (const atom of graph.atoms) {
    const symbol = atom.symbol === "CL" ? "Cl" : atom.symbol === "BR" ? "Br"
      : atom.symbol[0] + atom.symbol.slice(1).toLowerCase();
    counts.set(symbol, (counts.get(symbol) ?? 0) + 1);
    hydrogens += atom.hydrogens ?? 0;
  }
  if (hydrogens > 0) counts.set("H", (counts.get("H") ?? 0) + hydrogens);
  const symbols = [...counts.keys()];
  // Hill order: carbon, then hydrogen, then the rest alphabetically.
  const rest = symbols.filter((symbol) => symbol !== "C" && symbol !== "H").sort();
  const ordered = counts.has("C")
    ? ["C", ...(counts.has("H") ? ["H"] : []), ...rest]
    : symbols.sort();
  return ordered.map((symbol) =>
    symbol + (counts.get(symbol) === 1 ? "" : counts.get(symbol))).join("");
}
