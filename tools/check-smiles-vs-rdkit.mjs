#!/usr/bin/env node
/**
 * Does this port read a SMILES the way RDKit does? Formula, atoms, bonds, rings.
 *
 * 🔴 THE ONE GATE HERE THAT IS NOT THIS PORT COMPARED WITH ITSELF. Everything
 * else about SMILES in this repository - the unit tests, the round trips - is
 * a statement about internal consistency, and a parser can be consistently
 * wrong: the first version of this one gave caffeine C8H13N4O2 and ATP one
 * hydrogen too many, and every test it had still passed, because the bug was
 * in the rule and not in its application.
 *
 *   /home/ubuntu/.venv-rdkit/bin/python tools/oracle/dump_rdkit_smiles.py
 *   node tools/check-smiles-vs-rdkit.mjs
 *
 * 🔴 AND RDKit IS AN ORACLE, NOT A DEPENDENCY. Nothing under `src/` may import
 * it or anything derived from it; it does not ship to a browser. It is here to
 * be disagreed with in a way that names the atom.
 *
 * What is compared, and what deliberately is not:
 *
 *   formula     exact. The cheapest statement that the valence model is right.
 *   atoms       element, charge and hydrogen count, per atom, by INDEX - both
 *               sides keep the SMILES order, so index i is the same atom.
 *   bonds       which atoms are bonded: exact. And the integer ORDER, for
 *               every bond RDKit does not call aromatic.
 *   kekulé      🔴 AN AROMATIC BOND'S ORDER IS NOT COMPARED, BECAUSE IT IS NOT
 *               A FACT ABOUT THE MOLECULE. Benzene has two Kekulé structures
 *               and they are the same substance; which one a program picks
 *               falls out of the order it searched in. Asserted bond by bond,
 *               this failed six of fifty-one molecules - benzene, ibuprofen,
 *               paracetamol, ATP, porphine, NAD - and every difference was
 *               that arbitrary choice, with formula, hydrogens and rings all
 *               agreeing. What IS asserted is that the structure this port
 *               chose is a legal one: `valenceProblems` says every atom ends
 *               up with a valence it can have, which a wrong kekulisation
 *               breaks and a differently-chosen one does not.
 *   rings       the SSSR as sorted atom lists.
 *   fragments   how many disconnected pieces.
 *   aromaticity REPORTED, NOT ASSERTED. RDKit's aromaticity model is its own
 *               (it has five, selectable) and a lowercase atom in the input is
 *               not a promise about what any model will call aromatic
 *               afterwards - caffeine's carbonyl ring is the standard example,
 *               written aromatic and perceived non-aromatic. The kekulised
 *               ORDERS are what the geometry reads, and those are asserted.
 */

import { readFileSync } from "node:fs";
import { parseSmiles, molecularFormula } from "../src/chem/smiles.js";
import { smallestRings } from "../src/chem/rings.js";
import { valenceProblems } from "../src/chem/kekulize.js";

const DUMP = "oracle-dumps/rdkit-smiles.json";

const bondKey = (bond) => `${Math.min(bond.from, bond.to)}-${Math.max(bond.from, bond.to)}`;

function compare(record) {
  const problems = [];
  let graph;
  try {
    graph = parseSmiles(record.smiles);
  } catch (error) {
    return { problems: [`refused it: ${error.message}`], graph: null };
  }

  // RDKit writes a formula with a trailing charge (`C11H15N2O5+`); this port's
  // is the neutral formula, so the charge is compared separately through the
  // per-atom charges and stripped here rather than guessed at.
  const theirs = record.formula.replace(/[+-]\d*$/, "");
  const ours = molecularFormula(graph);
  if (ours !== theirs) problems.push(`formula ${ours} against RDKit's ${theirs}`);

  if (graph.atoms.length !== record.heavyAtoms) {
    problems.push(`${graph.atoms.length} heavy atoms against ${record.heavyAtoms}`);
  } else {
    graph.atoms.forEach((atom, index) => {
      const other = record.atoms[index];
      if (atom.symbol !== other.symbol) {
        problems.push(`atom ${index} is ${atom.symbol}, RDKit says ${other.symbol}`);
      }
      if (atom.charge !== other.charge) {
        problems.push(`atom ${index} ${atom.symbol} charge ${atom.charge} against ${other.charge}`);
      }
      if (atom.hydrogens !== other.hydrogens) {
        problems.push(`atom ${index} ${atom.symbol} has ${atom.hydrogens} H,`
          + ` RDKit says ${other.hydrogens}`);
      }
    });
  }

  const ourBonds = new Map(graph.bonds.map((bond) => [bondKey(bond), bond]));
  const theirBonds = new Map(record.bonds.map((bond) => [bondKey(bond), bond]));
  for (const [key, bond] of theirBonds) {
    const ourBond = ourBonds.get(key);
    if (ourBond === undefined) { problems.push(`bond ${key} missing`); continue; }
    // See the header: an aromatic bond's order is a choice of resonance form.
    // Its ORDER is checked by `valenceProblems` below; here only the fact of
    // the bond, and its order where RDKit says the bond is not aromatic.
    if (bond.aromatic) {
      if (ourBond.order !== 1 && ourBond.order !== 2) {
        problems.push(`aromatic bond ${key} kekulised to ${ourBond.order}`);
      }
      continue;
    }
    if (ourBond.order !== bond.order) {
      problems.push(`bond ${key} order ${ourBond.order} against ${bond.order}`);
    }
  }
  for (const problem of valenceProblems(graph)) {
    problems.push(`invalid kekulisation: ${problem}`);
  }
  for (const key of ourBonds.keys()) {
    if (!theirBonds.has(key)) problems.push(`bond ${key} is not RDKit's`);
  }

  const ourRings = smallestRings(graph).map((ring) => [...ring].sort((a, b) => a - b));
  const asText = (rings) => rings.map((ring) => ring.join(",")).sort().join(" | ");
  if (asText(ourRings) !== asText(record.rings)) {
    problems.push(`SSSR ${ourRings.length} rings [${asText(ourRings)}]`
      + ` against ${record.rings.length} [${asText(record.rings)}]`);
  }

  const fragments = new Set(graph.components).size;
  if (fragments !== record.fragments) {
    problems.push(`${fragments} fragments against ${record.fragments}`);
  }
  return { problems, graph };
}

function main() {
  let dump;
  try {
    dump = JSON.parse(readFileSync(DUMP, "utf8"));
  } catch {
    console.error(`no ${DUMP}. Generate it with:\n`
      + "  /home/ubuntu/.venv-rdkit/bin/python tools/oracle/dump_rdkit_smiles.py");
    process.exit(2);
  }

  const only = process.argv.find((argument) => argument.startsWith("--only="))?.slice(7);
  let failed = 0;
  let checked = 0;
  for (const record of dump.records) {
    if (record.error !== undefined) continue;
    if (only !== undefined && !record.name.includes(only)) continue;
    checked += 1;
    const { problems } = compare(record);
    if (problems.length === 0) {
      if (process.argv.includes("--verbose")) {
        console.log(`ok    ${record.name.padEnd(22)} ${record.formula}`);
      }
      continue;
    }
    failed += 1;
    console.log(`FAIL  ${record.name.padEnd(22)} ${record.smiles}`);
    for (const problem of problems.slice(0, 8)) console.log(`        ${problem}`);
    if (problems.length > 8) console.log(`        ...and ${problems.length - 8} more`);
  }

  console.log(`\n${checked - failed}/${checked} agree with RDKit ${dump.rdkit}`);
  if (failed > 0) process.exit(1);
}

main();
