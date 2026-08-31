import { describe, expect, it } from "./harness.js";
import { predictionToPdb, recyclesToPdb, safeJobName } from "../web/prediction-results.js";

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
});
