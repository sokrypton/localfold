import { readFileSync } from "node:fs";
import { describe, expect, it } from "./harness.js";
import {
  armOf, CONDITION_KINDS, inputStructurePdb, mergeConditions, residueRanges, segmentsOf, selectionOf,
} from "../web/condition-editor.js";
import { buildBatch } from "../shared/rfd4p/pipeline/index.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";

/**
 * The design page's condition editor (web/condition-editor.js) writes the condition spec a job
 * carries. tools/check-rfd4p-batch.py holds the pipeline's batch to upstream's for every condition
 * it offers, so what is held here is the editor's half: its rows must mean exactly what a
 * hand-written job means - the SAME BATCH, tensor for tensor - and it must refuse a value the
 * condition does not take, by name.
 */
const fixture = (name) => decompress(new Uint8Array(readFileSync(new URL(`../tools/fixtures/rcsb/${name}`, import.meta.url))));
const picks = (chain, residues) => residues.map((resId) => ({ chain, resId }));

describe("the condition editor", () => {
  it("writes residues as ranges and one sel() a chain", () => {
    expect(residueRanges([5, 3, 4, 9, 11, 10, 3])).toBe("3-5,9-11");
    expect(selectionOf([...picks("A", [16, 17, 18, 25]), ...picks("B", [377])], "od1, OD2"))
      .toBe("sel('A/*/[16-18,25]/[OD1,OD2]') or sel('B/*/[377]/[OD1,OD2]')");
    expect(selectionOf(picks("A", [3, 9]), "", "chain")).toBe("sel('A')");
  });

  it("refuses what a condition does not take, by name", () => {
    const refusal = (row) => { try { armOf(row); return ""; } catch (e) { return e.message; } };
    expect(refusal({ key: "C_MCOORD", value: "LIGAND", picks: picks("A", [1]) }))
      .toBe('Metal coordination takes METAL, COORDINATING, not "LIGAND"');
    expect(refusal({ key: "C_CENTROID", value: "1,2", picks: picks("A", [1]) }))
      .toBe('Chain centroid takes x,y,z in angstroms, not "1,2"');
    expect(refusal({ key: "C_SEQ", value: "true", picks: picks("A", [1]), atoms: "CA" }))
      .toBe("this condition applies to whole residues, so it takes no atom names");
    expect(refusal({ key: "C_HOT", value: "true", picks: [] })).toBe("pick residues in the viewer first");
    // every enum it offers is upstream's spelling, which the pipeline accepts (below, by building)
    expect(CONDITION_KINDS.length).toBe(15);
  });

  it("turns picked residues into From structure spans, one a run, a break between chains", () => {
    expect(JSON.stringify(segmentsOf([...picks("A", [18, 16, 17, 30]), ...picks("B", [377]), ...picks("A", [-3, -2])])))
      .toBe(JSON.stringify([
        { kind: "segment", chain: "A", first: -3, last: -2, hotspots: "" },
        { kind: "segment", chain: "A", first: 16, last: 18, hotspots: "" },
        { kind: "segment", chain: "A", first: 30, last: 30, hotspots: "" },
        { kind: "break" },
        { kind: "segment", chain: "B", first: 377, last: 377, hotspots: "" },
      ]));
  });

  it("appends its rows after a preset's, in order, so the last set wins", () => {
    const merged = mergeConditions({ C_CRD: { true: [{ select: "sel('A')" }] } },
      [{ key: "C_CRD", value: "false", picks: picks("A", [3]) },
        { key: "C_IDX", value: "loop1", picks: picks("A", [5, 6]) }]);
    expect(JSON.stringify(merged)).toBe(JSON.stringify({
      C_CRD: { true: [{ select: "sel('A')" }], false: [{ select: "sel('A/*/[3]')" }] },
      C_IDX: { false: [{ select: "sel('A/*/[5-6]')", group: "loop1" }] },
    }));
  });

  it("builds the batch a hand-written job builds, condition for condition", () => {
    const cif = fixture("6mrr.cif.zst");
    const lys = [8, 14, 18, 19, 46, 55, 58, 59, 66];
    const glu = [5, 7, 11, 12, 15, 20, 26];
    const rows = [
      { key: "C_CRD", value: "true", picks: picks("A", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) },
      { key: "C_RAS", value: "BURIED", picks: picks("A", [16, 17]) },
      { key: "C_RAS", value: "EXPOSED", picks: picks("A", [30, 31, 32]) },
      { key: "C_SALT", value: "POSITIVE", picks: picks("A", lys), atoms: "NZ" },
      { key: "C_SALT", value: "NEGATIVE", picks: picks("A", glu), atoms: "OE1,OE2" },
      { key: "C_HB", value: "DONOR", picks: picks("A", [40]), atoms: "N" },
      { key: "C_HPHOB", value: "true", picks: picks("A", [6, 13, 17]), atoms: "CD1,CD2" },
      { key: "C_SEC", value: "HELIX", picks: picks("A", [44, 45, 46, 47]) },
      { key: "C_CENTROID", value: "1.5,-2,3", picks: picks("A", [9]) },
      { key: "C_CATH", value: "1.10.x", picks: picks("A", Array.from({ length: 68 }, (_, i) => i + 1)) },
    ];
    const byHand = {
      C_CRD: { true: [{ select: "sel('A/*/[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]')" }] },
      C_RAS: { BURIED: [{ select: "sel('A/*/16') or sel('A/*/17')" }], EXPOSED: [{ select: "sel('A/*/[30-32]')" }] },
      C_SALT: { POSITIVE: [{ select: "sel('A/LYS/*/NZ')" }], NEGATIVE: [{ select: `sel('A/GLU/[${glu}]/[OE1,OE2]')` }] },
      C_HB: { DONOR: [{ select: "sel('A/*/40/N')" }] },
      C_HPHOB: { true: [{ select: "sel('A/LEU/[6,13,17]/[CD1,CD2]')" }] },
      C_SEC: { HELIX: [{ select: "sel('A/*/[44-47]')" }] },
      C_CENTROID: { "1.5,-2,3": [{ select: "sel('A')" }] },
      C_CATH: { "1.10.x": [{ select: "sel('A')" }] },
    };
    const build = (conditions) => buildBatch({ contig: "A1-68", conditions }, { inputCif: cif, seed: 0 }).batch;
    const fromEditor = build(mergeConditions(undefined, rows));
    const fromHand = build(byHand);
    const keys = Object.keys(fromHand).sort();
    expect(Object.keys(fromEditor).sort().join()).toBe(keys.join());
    const differ = keys.filter((k) => {
      const a = fromEditor[k].data;
      const b = fromHand[k].data;
      return a.length !== b.length || a.some((v, i) => !Object.is(v, b[i]));
    });
    expect(differ.join()).toBe("");
    // ...and the conditions arrived: a test of two empty batches would pass too.
    for (const key of ["condition_saltbridge_1_atom", "condition_rasa-binned_1_atom", "condition_hbond_1_atom",
      "condition_hydrophobic_1_atom", "condition_secondary-structure_1_residue"]) {
      expect(`${key} ${fromEditor[key].data.some((v) => v !== 0)}`).toBe(`${key} true`);
    }
  });

  it("draws the structure in the selection language's names", () => {
    const atoms = parseCifToAtomArray(fixture("1prw.cif.zst"));
    const { pdb, chainOf } = inputStructurePdb(atoms);
    // label chains B-E are the four calcium ions; waters (F) are not drawn
    expect([...chainOf.values()].join()).toBe("A,B,C,D,E");
    const ca = pdb.split("\n").filter((l) => l.startsWith("HETATM") && l.slice(17, 20) === " CA");
    expect(ca.map((l) => `${l[21]}${l.slice(22, 26).trim()}`).join()).toBe("B377,C378,D379,E380");
    expect(pdb.includes("HOH")).toBe(false);
  });
});
