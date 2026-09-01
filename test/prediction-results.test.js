import { describe, expect, it } from "./harness.js";
import { confidenceJson, predictionToPdb, recyclesToPdb, safeJobName } from "../web/prediction-results.js";

describe("browser prediction result formatting", () => {
  it("writes only present atom37 coordinates and pLDDT B-factors", () => {
    const atom37 = new Float32Array(37 * 3);
    atom37.set([1.25, -2.5, 3.75], 0);
    const atom37Mask = new Float32Array(37); atom37Mask[0] = 1;
    const pdb = predictionToPdb("A", {
      atom14: new Float32Array(), atom37, atom37Mask, finalRepresentation: new Float32Array(),
      affine: new Float32Array(), angles: new Float32Array(), unnormalizedAngles: new Float32Array(),
      elapsedMilliseconds: 0,
    }, Float32Array.of(97.25));
    expect(pdb).toContain("ATOM      1  N   ALA A   1");
    expect(pdb).toContain("   1.250  -2.500   3.750  1.00 97.25");
    expect(pdb.endsWith("TER\nEND\n")).toBe(true);
  });

  it("starts a one-character element's atom name in column 14", () => {
    // 🔴 THE COLUMNS, NOT THE TRIMMED TEXT. The PDB format gives the atom name
    // columns 13-16 and starts a one-character element's name at 14 - " CA " -
    // reserving column 13 for a two-character ELEMENT like iron. This used to
    // be padStart(4), which right-justified it into "  CA": every lenient
    // parser trims that back to the right name, and every strict one reads by
    // column and does not find the backbone where N, CA and C belong. What it
    // looks like downstream is a structure drawn with no backbone.
    const atom37 = new Float32Array(37 * 3);
    const atom37Mask = new Float32Array(37);
    // ATOM_NAMES order: N, CA, C, CB, O, CG, CG1 - so this covers a one, two
    // and three character name, which is every width a protein atom has.
    for (const slot of [0, 1, 2, 3, 6]) atom37Mask[slot] = 1;
    const pdb = predictionToPdb("A", { atom37, atom37Mask }, Float32Array.of(50));
    const names = pdb.split("\n").filter((line) => line.startsWith("ATOM"))
      .map((line) => line.slice(12, 16));
    expect(names).toEqual([" N  ", " CA ", " C  ", " CB ", " CG1"]);
  });

  it("writes each recycle as its own model, carrying that pass's pLDDT", () => {
    const frame = (x, plddt) => {
      const atom37 = new Float32Array(37 * 3);
      atom37.set([x, 0, 0], 0);
      const atom37Mask = new Float32Array(37); atom37Mask[0] = 1;
      return { structure: { atom37, atom37Mask }, confidence: { plddt: Float32Array.of(plddt) } };
    };
    const pdb = recyclesToPdb("A", [frame(1, 40), frame(2, 70), frame(3, 95)]);
    expect(pdb.match(/^MODEL /gm).length).toBe(3);
    expect(pdb.match(/^ENDMDL$/gm).length).toBe(3);
    // ...the coordinate and the confidence both move, which is what animates.
    expect(pdb).toContain("   1.000   0.000   0.000  1.00 40.00");
    expect(pdb).toContain("   3.000   0.000   0.000  1.00 95.00");
    expect(pdb.endsWith("ENDMDL\nEND\n")).toBe(true);
  });

  it("writes oligomers as separate PDB chains with numbering restarted", () => {
    const atom37 = new Float32Array(3 * 37 * 3);
    const atom37Mask = new Float32Array(3 * 37);
    atom37Mask[0] = 1; atom37Mask[37] = 1; atom37Mask[74] = 1;
    const pdb = predictionToPdb("ACD", { atom37, atom37Mask }, Float32Array.of(90, 80, 70), [1, 2]);
    expect(pdb).toContain(" ALA A   1");
    expect(pdb).toContain(" CYS B   1");
    expect(pdb).toContain(" ASP B   2");
    expect(pdb.match(/^TER$/gm).length).toBe(2);
  });

  it("refuses an empty recycle list", () => {
    expect(() => recyclesToPdb("A", [])).toThrow(/at least one recycle/);
  });

  it("makes download names safe", () => {
    expect(safeJobName(" ../../my fold ")).toBe("my_fold");
    expect(safeJobName("***")).toBe("prediction");
  });

  it("formats confidence JSON with pLDDT, pTM, and multi-chain ipTM", () => {
    const plddt = Float32Array.of(90, 85);
    const pae = Float32Array.of(1, 5, 5, 1);
    const jsonStr = confidenceJson("AC", {
      plddt,
      meanPlddt: 87.5,
      ptm: 0.82,
      iptm: 0.75,
      multimerScore: 0.764,
      predictedAlignedError: pae,
      maxPredictedAlignedError: 31.75,
    });
    const parsed = JSON.parse(jsonStr);
    expect(parsed.mean_plddt).toBe(87.5);
    expect(parsed.ptm).toBe(0.82);
    expect(parsed.iptm).toBe(0.75);
    expect(parsed.ranking_confidence).toBeCloseTo(0.764, 3);
    expect(parsed.predicted_aligned_error).toEqual([[1, 5], [5, 1]]);
  });
});

const { computeTmScores: computeTmScoresPinned } =
  await import("../src/heads/confidence.js");

describe("TM and interface TM score calculation", () => {
  it("computes pTM and multi-chain ipTM from PAE logits", async() => {
    const { computeTmScores } = await import("../src/heads/confidence.js");
    const length = 20;
    const bins = 64;
    const breaks = Float32Array.from({ length: 63 }, (_, i) => i * 0.5);
    const logits = new Float32Array(length * length * bins);

    // Make diagonal/intra-chain pairs (0..9, 0..9 and 10..19, 10..19) have high confidence in low PAE bin 0
    // Make cross-chain pairs have confidence in bin 6
    for (let i = 0; i < length; i++) {
      for (let j = 0; j < length; j++) {
        const isSame = (i < 10 && j < 10) || (i >= 10 && j >= 10);
        const preferredBin = isSame ? 0 : 6;
        logits[(i * length + j) * bins + preferredBin] = 10.0;
      }
    }

    const monomerScores = computeTmScores(logits, length, breaks);
    expect(monomerScores.ptm).toBeGreaterThan(0.2);
    expect(monomerScores.iptm).toBe(undefined);

    const multimerScores = computeTmScores(logits, length, breaks, [10, 10]);
    expect(multimerScores.ptm).toBeCloseTo(monomerScores.ptm, 4);
    expect(typeof multimerScores.iptm).toBe("number");
    expect(multimerScores.iptm).toBeGreaterThan(0.005);
    expect(multimerScores.multimerScore).toBeCloseTo(0.8 * multimerScores.iptm + 0.2 * multimerScores.ptm, 4);
  });

  it("holds AlphaFold 2's scores to the exact values it produces today", () => {
    // 🔴 A CHARACTERISATION TEST, PINNED BEFORE THE REDUCTION WAS SHARED WITH
    // AF3. The assertions above are all `greaterThan`, which is enough to say
    // the function does something and not enough to say it still does the same
    // thing. These numbers are a record of behaviour, not a claim about
    // correctness.
    //
    // 🔴 AND SHARING IT MOVED THEM, BY 2e-10. The old code built its TM-per-bin
    // table with `centers.map(...)`, and mapping a Float32Array returns a
    // Float32Array, so the table was single precision; tmPerBinFor returns
    // float64. That is the whole of the difference - the values below are the
    // post-refactor ones and the pre-refactor ptm was 0.31470439840500297
    // against 0.31470439859116128 now. It is ten orders below anything this
    // score is read to, and it is written down rather than rounded away because
    // the next difference this test catches might not be.
    const length = 20;
    const bins = 64;
    const breaks = Float32Array.from({ length: 63 }, (_, i) => i * 0.5);
    const logits = new Float32Array(length * length * bins);
    for (let i = 0; i < length; i += 1) {
      for (let j = 0; j < length; j += 1) {
        const isSame = (i < 10 && j < 10) || (i >= 10 && j >= 10);
        logits[(i * length + j) * bins + (isSame ? 0 : 6)] = 10.0;
      }
    }
    const monomer = computeTmScoresPinned(logits, length, breaks);
    const multimer = computeTmScoresPinned(logits, length, breaks, [10, 10]);
    expect(monomer.ptm).toBeCloseTo(0.314704398591, 12);
    expect(monomer.iptm).toBe(undefined);
    expect(multimer.ptm).toBeCloseTo(0.314704398591, 12);
    expect(multimer.iptm).toBeCloseTo(0.009639396126, 12);
    expect(multimer.multimerScore).toBeCloseTo(0.070652396619, 12);
  });
});
