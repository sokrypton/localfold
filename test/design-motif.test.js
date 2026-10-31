import { test } from "node:test";
import assert from "node:assert/strict";
import { heldCoordinates, motifMatch, motifTerms } from "../cpu/design/motif.js";
import { PROTOCOLS, protocolFor, verdict } from "../cpu/design/protocol.js";
import { search } from "../web/search.js";

/** A loaded structure as the page holds one (parallel arrays), from [chain, residue, resname, atom, element, xyz]. */
function structure(rows) {
  return {
    chain_id: rows.map((r) => r[0]), res_id: rows.map((r) => r[1]), res_name: rows.map((r) => r[2]),
    atom_name: rows.map((r) => r[3]), element: rows.map((r) => r[4]), coord: rows.map((r) => r[5]),
  };
}

const loaded = structure([
  ["M", 10, "HIS", "CA", "C", [0, 0, 0]], ["M", 10, "HIS", "NE2", "N", [1, 0, 0]], ["M", 10, "HIS", "H", "H", [9, 9, 9]],
  ["M", 11, "ASP", "CG", "C", [0, 2, 0]], ["M", 11, "ASP", "OD1", "O", [0, 3, 0]],
  ["D", 1, "DA", "P", "P", [5, 5, 5]], ["D", 1, "DA", "C1'", "C", [5, 6, 5]], ["D", 2, "DT", "P", "P", [6, 5, 5]],
  ["L", 1, "ATP", "PA", "P", [3, 3, 3]], ["L", 1, "ATP", "C1'", "C", [3, 4, 3]],
]);
const parts = [
  { index: 0, at: 5, chain: "M", first: 10, last: 11, held: true, polymer: "protein", designed: true },
  { index: 1, at: 0, chain: "D", first: 1, last: 2, held: true, polymer: "dna", designed: false },
];

test("held coordinates: the motif at its positions, a strand after the designed chain, a ligand by its chain", () => {
  const held = heldCoordinates({ atoms: loaded, chains: ["XXXXXHDXXX", "AT"], chainIndex: 0, ligandCodes: ["ATP"], structureParts: parts });
  assert.deepEqual(held.motif.map((m) => [m.position, m.res, m.atoms.map((a) => a[0]).join(",")]),
    [[5, "HIS", "CA,NE2"], [6, "ASP", "CG,OD1"]], "hydrogens are not guided");
  const entries = held.atoms(10);
  assert.deepEqual(entries.map((e) => [e.token, e.atom]), [
    [5, "CA"], [5, "NE2"], [6, "CG"], [6, "OD1"], [10, "P"], [10, "C1'"], [11, "P"], [-3, "PA"], [-3, "C1'"]]);
  assert.ok(entries.every((e) => e.optional === true));
  // a wholly designed chain is the run's length, which moves the strand's tokens
  const free = heldCoordinates({ atoms: loaded, chains: ["", "AT"], chainIndex: 0, ligandCodes: [], structureParts: [parts[1]] });
  assert.deepEqual(free.atoms(40).map((e) => e.token), [40, 40, 41]);
  assert.throws(() => heldCoordinates({ atoms: loaded, chains: ["", "AT"], chainIndex: 0, ligandCodes: ["SAM"] }), /SAM .* not in the loaded structure/);
});

const pdbLine = (serial, name, res, chain, residue, [x, y, z]) =>
  `ATOM  ${String(serial).padStart(5)} ${name.padEnd(4)} ${res} ${chain}${String(residue).padStart(4)}    `
  + `${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}  1.00  0.00           ${name[0]}`;

/** Chain A of `n` ALA residues with HIS and ASP at `at`, their tips laid as the motif; a ligand atom on chain B. */
function designPdb(n, at, { bend = 0, ligandAt = [2, 2, 1] } = {}) {
  const rows = [];
  let serial = 1;
  for (let r = 1; r <= n; r += 1) {
    const base = [r * 3.8, 10, 0];
    const res = r - 1 === at[0] ? "HIS" : r - 1 === at[1] ? "ASP" : "ALA";
    for (const [k, name] of ["N", "CA", "C", "O"].entries()) rows.push(pdbLine(serial++, name, res, "A", r, [base[0] + k * 0.4, base[1], k * 0.3]));
    if (res === "HIS") { rows.push(pdbLine(serial++, "CG", res, "A", r, [0, 0, 0])); rows.push(pdbLine(serial++, "NE2", res, "A", r, [1, 0, 0])); }
    if (res === "ASP") { rows.push(pdbLine(serial++, "CG", res, "A", r, [0, 2, bend])); rows.push(pdbLine(serial++, "OD1", res, "A", r, [0, 3, 0])); }
  }
  rows.push(pdbLine(serial++, "C1", "LIG", "B", 1, ligandAt).replace("ATOM  ", "HETATM"));
  return rows.join("\n") + "\n";
}
const motif = [
  { position: 3, res: "HIS", atoms: [["CG", 0, 0, 0], ["NE2", 1, 0, 0]] },
  { position: 7, res: "ASP", atoms: [["CG", 0, 2, 0], ["OD1", 0, 3, 0]] },
];

test("motif terms: exact at the motif's positions, off when it is bent, and a clash counted in the design", () => {
  const pdb = designPdb(12, [3, 7]);
  const exact = motifTerms(motif, [3, 7], pdb, pdb);
  assert.ok(exact.motifRmsd < 1e-3 && exact.clashes === 0, JSON.stringify(exact));
  assert.ok(motifTerms(motif, [3, 7], pdb, designPdb(12, [3, 7], { bend: 2 })).motifRmsd > 0.5);
  assert.ok(motifTerms(motif, [3, 7], designPdb(12, [3, 7], { ligandAt: [0.5, 0.2, 0] }), pdb).clashes > 0);
});

test("the motif is found unindexed, wherever the fold put it", () => {
  const match = motifMatch(motif, designPdb(14, [9, 2]));
  assert.deepEqual(match.positions, [9, 2]);
  assert.ok(match.rmsd < 1e-3);
  assert.equal(motifMatch([{ res: "TRP", atoms: [["CA", 0, 0, 0]] }], designPdb(14, [9, 2])), null);
});

test("fixed and AME recipes follow the job: Hold coordinates, and a motif only beside a ligand", () => {
  assert.equal(protocolFor({ partners: 1, ligands: 1, fixed: true }).protocol, PROTOCOLS.smFixed);
  assert.equal(protocolFor({ partners: 2, nucleic: 2, fixed: true }).protocol, PROTOCOLS.dnaFixed);
  assert.equal(protocolFor({ partners: 1, ligands: 1, motif: 2, fixed: true }).protocol, PROTOCOLS.ame);
  assert.match(protocolFor({ partners: 1, ligands: 1, motif: 2 }).why, /Hold coordinates/);
  assert.match(protocolFor({ motif: 2, fixed: true }).why, /no ligand/);
  const terms = { ipae: 1.0 };
  const pdb = designPdb(12, [3, 7]);
  assert.ok(verdict("ame", pdb, pdb, terms, "A", { motif, positions: [3, 7] }).pass);
  assert.ok(!verdict("ame", pdb, designPdb(12, [3, 7], { bend: 8 }), terms, "A", { motif, positions: [3, 7] }).pass);
});

const folded = (ipae, pdb) => ({ pdb, batch: { asymId: Int32Array.from([1, 2]) },
  confidence: { predictedAlignedError: Float32Array.from([0, ipae, ipae, 0]), chainPtm: { 1: 0.9, 2: 0.9 }, chainPairIptm: { "1|2": 0.9 } } });

test("AME: the guide reaches the hunt, every cycle is judged by its unguided probe, and the climb runs on a miss", async () => {
  const hunts = [];
  const folds = [];
  let draws = 0;
  const held = { atoms: () => [{ token: 3, atom: "CG", xyz: [0, 0, 0] }], motif };
  const out = await search({ chains: ["", "LIG"], seed: 1, held,
    protocol: { ...PROTOCOLS.ame, restarts: 1, climb: { from: 1, gens: 1, draws: 2, top: 1 } } }, {
    hunt: async (options) => {
      hunts.push({ until: options.motif?.until, atoms: options.motif?.atoms, examine: typeof options.examine });
      for (const cycle of [0, 1, 2]) {
        // the guided design fold holds the motif; the probe (an unguided fold of the same sequence) bends it
        const guided = folded(1.0, designPdb(12, [3, 7]));
        const examined = await options.examine({ sequence: `S${cycle}:LIG`, folded: guided, context: { cycle } });
        options.report({ cycle, score: 0.9, alanine: 0.05, vetoed: null, sequence: `S${cycle}`, pdb: guided.pdb,
          folded: { ...guided, examined } });
      }
      return { best: null, cycles: [] };
    },
    fold: async (sequence, { steps }) => {
      folds.push(steps);
      return folded(1.0, designPdb(12, [3, 7], { bend: 8 }));
    },
    redesign: async () => { draws += 1; return { sequence: `C${draws}` }; },
  });
  assert.equal(hunts[0].until, 1.0);
  assert.equal(hunts[0].atoms, held.atoms);
  assert.equal(hunts[0].examine, "function");
  assert.equal(folds.filter((s) => s === 100).length, 3 + 2, "three probes, then two climb draws");
  assert.equal(out.climbed, 2);
  assert.equal(out.passed, false);
  assert.ok(out.best.motifRmsd > 0.5, "judged by the bent probe, not the guided fold");
  assert.ok(out.records.filter((r) => r.restart === 0).every((r) => r.motifRmsd > 0.5));
});

test("a guided recipe with nothing to hold is refused before anything folds", async () => {
  await assert.rejects(search({ chains: ["", "AT"], protocol: PROTOCOLS.dnaFixed }, { hunt: async () => ({}) }), /holds coordinates/);
});
