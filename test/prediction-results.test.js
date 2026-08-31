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
    expect(pdb).toContain("ATOM      1    N ALA A   1");
    expect(pdb).toContain("   1.250  -2.500   3.750  1.00 97.25");
    expect(pdb.endsWith("TER\nEND\n")).toBe(true);
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
});
