/**
 * shared/rfd4p/symmetry.js's automorphisms of each component on stdin, for tools/check-rfd4p-automorphisms.py.
 *
 *     echo '[{"element": [...], "charge": [...], "bonds": [[i, j, type], ...]}]' | node tools/rfd4p-automorphisms.mjs
 *
 * Each component is one unit coloured as atomworks colours it (element and heavy-atom degree,
 * edges by the resonance-normalised order); the output is each one's permutations, identity first.
 */
import { readFileSync } from "node:fs";
import { automorphisms, heavyNeighbours, normalizedBondOrders } from "../shared/rfd4p/symmetry.js";

const cases = JSON.parse(readFileSync(0, "utf8"));
const out = cases.map(({ element, charge, bonds }) => {
  const heavy = heavyNeighbours({ n: element.length, element, bonds });
  const orders = normalizedBondOrders({ element, charge, bonds }, heavy);
  const colour = element.map((e, i) => `${e}/${heavy[i]}`);
  return automorphisms(colour, bonds.map(([i, j], k) => [i, j, orders[k]])).map((p) => Array.from(p));
});
process.stdout.write(JSON.stringify(out));
