import { test } from "node:test";
import assert from "node:assert/strict";
import { PROTOCOLS, confident, footprint, interfaceTerms, protocolFor, verdict } from "../cpu/design/protocol.js";
import { search } from "../web/search.js";

const line = (record, serial, name, residue, chain, [x, y, z], element) =>
  `${record.padEnd(6)}${String(serial).padStart(5)} ${name.padEnd(4)} ${record === "ATOM" ? "ALA" : "LIG"} ${chain}`
  + `${String(residue).padStart(4)}    ${x.toFixed(3).padStart(8)}${y.toFixed(3).padStart(8)}${z.toFixed(3).padStart(8)}`
  + `  1.00  0.00          ${element.padStart(2)}`;

/** A small complex: a 6-residue binder on chain A, a 6-residue partner on B (or a 4-atom ligand), `shift` moving A. */
function complex({ shift = [0, 0, 0], ligand = false, turn = 0 } = {}) {
  const rows = [];
  let serial = 1;
  const c = Math.cos(turn), s = Math.sin(turn);
  const at = ([x, y, z]) => [c * x - s * y, s * x + c * y, z];
  for (let r = 1; r <= 6; r += 1) {
    for (const [k, name] of ["N", "CA", "C", "O"].entries()) {
      rows.push(line("ATOM", serial++, name, r, "A", at([r * 3.8 + k * 0.5 + shift[0], 4 + k * 0.3 + shift[1], (r % 2) * 1.1 + shift[2]]), name[0]));
    }
  }
  if (ligand) {
    for (let k = 0; k < 4; k += 1) rows.push(line("HETATM", serial++, `C${k + 1}`, 1, "B", at([8 + k * 1.4, 1, k * 0.4]), "C"));
  } else {
    for (let r = 1; r <= 6; r += 1) {
      for (const [k, name] of ["N", "CA", "C", "O"].entries()) {
        rows.push(line("ATOM", serial++, name, r, "B", at([r * 3.8 + k * 0.5, -0.5 + k * 0.2, (r % 3) * 0.9]), name[0]));
      }
    }
  }
  return rows.join("\n") + "\n";
}

test("the recipe follows what the job carries", () => {
  assert.equal(protocolFor({ partners: 1 }).protocol, PROTOCOLS.ppi);
  assert.equal(protocolFor({ partners: 1, ligands: 1 }).protocol, PROTOCOLS.sm);
  assert.equal(protocolFor({ partners: 2, nucleic: 2 }).protocol, PROTOCOLS.dna);
  assert.equal(protocolFor({}).protocol, null);
  assert.equal(protocolFor({ ligands: 1, nucleic: 2 }).protocol, null);
  assert.equal(protocolFor({ partners: 1, designs: "dna" }).protocol, null);
});

test("interface terms: the designed chain's pTM, its mean pair ipTM and the lowest two-way interface PAE", () => {
  const fold = {
    batch: { asymId: Int32Array.from([1, 1, 2, 3]) },
    confidence: {
      // rows: from token, columns: to token
      predictedAlignedError: Float32Array.from([0, 0, 3, 9, 0, 0, 1, 8, 2, 4, 0, 0, 7, 6, 0, 0]),
      chainPtm: { 1: 0.9, 2: 0.7, 3: 0.5 },
      chainPairIptm: { "1|2": 0.8, "1|3": 0.4, "2|3": 0.1 },
    },
  };
  const terms = interfaceTerms(fold, 0);
  assert.equal(terms.binderPtm, 0.9);
  assert.ok(Math.abs(terms.iptm - 0.6) < 1e-12);
  assert.equal(terms.ipae, 1.5); // chain 2: min 1 one way, min 2 the other
  assert.ok(confident(terms, { confidence: 0.85, ipae: 1.5 }, "ppi"));
  assert.ok(!confident(terms, { confidence: 0.85, ipae: 1.5 }, "sm"));
  assert.ok(confident({ ipae: 1, binderPtm: 0, iptm: 0 }, { confidence: 0.85, ipae: 1.2 }, "dna"));
});

test("criteria: a rigidly moved refold passes and a displaced binder fails, by the paper's thresholds", () => {
  const terms = { ipae: 1.2, binderPtm: 0.85, iptm: 0.85 };
  const design = complex();
  const same = verdict("ppi", design, complex({ turn: 0.9 }), terms);
  assert.ok(same.rmsd < 1e-3 && same.pass, JSON.stringify(same));
  const slid = verdict("ppi", design, complex({ shift: [3, 0, 0] }), terms);
  assert.ok(Math.abs(slid.rmsd - 3) < 1e-3 && !slid.pass && slid.margin > 1, JSON.stringify(slid));
  assert.ok(verdict("dna", design, complex({ shift: [3, 0, 0] }), terms).pass, "DNA allows 8 A");
  assert.ok(!verdict("ppi", design, design, { ...terms, binderPtm: 0.7 }).pass, "confidence is part of it");
  const ligand = complex({ ligand: true });
  const held = verdict("sm", ligand, complex({ ligand: true, turn: 2 }), terms);
  assert.ok(held.pass && held.ligandRmsd < 1e-3, JSON.stringify(held));
  const away = verdict("sm", ligand, complex({ ligand: true, shift: [6, 0, 0] }), terms);
  assert.ok(away.ligandRmsd > 5 && !away.pass, JSON.stringify(away));
});

test("the footprint counts partner residues within 4.5 A of the designed chain", () => {
  assert.equal(footprint(complex()), 5);
  assert.equal(footprint(complex({ shift: [0, 30, 0] })), 0);
});

/** A fold result whose interface terms are `ipae` and `ptm`, over the small complex. */
const folded = (ipae, ptm, pdb = complex()) => ({
  pdb,
  batch: { asymId: Int32Array.from([1, 2]) },
  confidence: { predictedAlignedError: Float32Array.from([0, ipae, ipae, 0]), chainPtm: { 1: ptm, 2: 0.9 }, chainPairIptm: { "1|2": ptm } },
});

/** A scripted hunt: restart r's cycles get the (ipae, ptm) pairs `script[r]`, aborting as a real hunt would. */
function scriptedHunt(script, calls) {
  return async (options) => {
    const restart = (options.seed - 5) / 101;
    calls.push({ restart, cycles: options.cycles, omit: options.omit, designer: options.designer, foldTries: options.foldTries });
    for (const [cycle, [ipae, ptm]] of (script[restart] ?? []).entries()) {
      if (options.signal.aborted) throw new DOMException("stopped", "AbortError");
      options.report({ cycle, score: ptm, alanine: 0.05, vetoed: null, sequence: `SEQ${restart}${cycle}`, pdb: complex(), folded: folded(ipae, ptm) });
    }
    return { best: null, cycles: [] };
  };
}

const base = { chains: ["", "PARTNER"], seed: 5 };

test("a confident cycle stops its restart, its check passes and the search ends there", async () => {
  const calls = [];
  const checks = [];
  const out = await search({ ...base, protocol: PROTOCOLS.ppi, onCheck: (c) => checks.push(c) }, {
    hunt: scriptedHunt([[[20, 0.2], [2.0, 0.82], [1.1, 0.9], [1.0, 0.95]], [[1, 0.9]]], calls),
    fold: async (sequence, { seed, steps, recycles }) => {
      assert.equal(steps, 200);
      assert.equal(recycles, 5);
      assert.equal(sequence, "SEQ02:PARTNER");
      assert.equal(seed, 5 * 7 + 2);
      return folded(1.0, 0.9);
    },
  });
  assert.deepEqual(calls.map((c) => c.restart), [0], "one restart: its check passed");
  assert.equal(calls[0].cycles, 6);
  assert.equal(out.passed, true);
  assert.equal(out.best.sequence, "SEQ02");
  assert.equal(checks.length, 1, "the best cycle checked first, and it passed");
  assert.equal(out.records.length, 3, "the stop aborted the hunt after cycle 2");
});

test("nothing passing: every restart runs, the polish redraws the nearest miss, and the nearest is returned", async () => {
  const calls = [];
  let draws = 0;
  const out = await search({ ...base, protocol: { ...PROTOCOLS.dna, restarts: 2, minFootprint: 4, polish: { from: 1, draws: 2, top: 1 } } }, {
    hunt: scriptedHunt([[[20, 0.2], [2.0, 0.5]], [[20, 0.2], [2.2, 0.5]]], calls),
    // every check refold puts the binder 9 A away: confident, and too far for DNA's 8 A
    fold: async (sequence, { steps }) => (steps === 200 ? folded(1.0, 0.9, complex({ shift: [9, 0, 0] })) : folded(1.9, 0.5)),
    redesign: async (pdb, { temperature, omit }) => {
      draws += 1;
      assert.equal(temperature, 0.1);
      assert.equal(omit, "CDE");
      return { sequence: `POLISH${draws}` };
    },
  });
  assert.deepEqual(calls.map((c) => [c.restart, c.omit, c.designer, c.foldTries]), [[0, "CDE", "ligand", 5], [1, "CDE", "ligand", 5]]);
  assert.equal(out.passed, false);
  assert.equal(out.polished, 2);
  assert.equal(out.checks.length, 3, "a check per restart, and one of the polished draws");
  assert.ok(out.best.check !== undefined && !out.best.check.pass, "the nearest checked miss");
});

test("a search stopped by its caller ends, rather than being read as an early stop", async () => {
  const stop = new AbortController();
  stop.abort();
  await assert.rejects(search({ ...base, protocol: PROTOCOLS.ppi, signal: stop.signal }, {
    hunt: scriptedHunt([[[20, 0.2], [1.0, 0.95]]], []), fold: async () => folded(1, 0.9),
  }), /stopped/);
});
