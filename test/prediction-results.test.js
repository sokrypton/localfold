import { describe, expect, it } from "./harness.js";
import { confidenceJson, matrixForViewer, modifiedPositions, predictionToPdb, recyclesToPdb,
  safeJobName, viewerTokens } from "../web/prediction-results.js";
import { featuriseProtein } from "../src/af3/featurise/featurise.js";
import { toPdb } from "../src/af3/fold.js";

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

describe("a token matrix in the viewer's index space", () => {
  // A twelve-residue chain with a ten-atom phosphoserine at position 3, which
  // is the shape measured against the real viewer: AF3 makes 21 tokens of it
  // and py2Dmol draws 12 positions.
  const SEP = ["N", "CA", "C", "O", "CB", "OG", "P", "O1P", "O2P", "O3P"];
  const chainWithSep = () => ({
    tokens: 21,
    modifiedSpans: [{ from: 2, count: SEP.length, code: "SEP", residue: 2,
                      atoms: SEP.map((name) => ({ name })) }],
  });

  it("keeps one token per residue and picks the atom the viewer draws", () => {
    const keep = viewerTokens(chainWithSep());
    // 0, 1, then the phosphoserine's CA (token 3), then 12..20
    expect(keep.length).toBe(12);
    expect(keep.slice(0, 4)).toEqual([0, 1, 3, 12]);
    expect(keep[11]).toBe(20);
  });

  it("leaves a fold with nothing atomised exactly as it was", () => {
    const keep = viewerTokens({ tokens: 5, modifiedSpans: [] });
    expect(keep).toEqual([0, 1, 2, 3, 4]);
  });

  it("does not collapse a ligand, whose atoms are positions too", () => {
    // Six heavy atoms of a glycerol after a ten-residue chain: sixteen tokens
    // and sixteen positions, which is what the viewer already agreed with.
    const keep = viewerTokens({ tokens: 16, modifiedSpans: [],
                                ligandSpans: [{ from: 10, count: 6, code: "GOL" }] });
    expect(keep.length).toBe(16);
  });

  it("keeps boltz2's one-token modification as one token", () => {
    const keep = viewerTokens({ tokens: 12, modifiedSpans: [
      { from: 2, count: 1, code: "SEP", residue: 2, oneToken: true,
        atoms: SEP.map((name) => ({ name })) }] });
    expect(keep.length).toBe(12);
  });

  it("reads the rows and columns the kept tokens name", () => {
    // value(i, j) = i * 100 + j, so a misread is legible in the number itself
    const stride = 21;
    const values = new Float32Array(stride * stride);
    for (let row = 0; row < stride; row += 1) {
      for (let col = 0; col < stride; col += 1) values[row * stride + col] = row * 100 + col;
    }
    const keep = viewerTokens(chainWithSep());
    const rows = matrixForViewer(values, keep);
    expect(rows.length).toBe(12);
    expect(rows[0].length).toBe(12);
    // the residue after the modification is token 12, not token 3
    expect(rows[3][3]).toBe(12 * 100 + 12);
    // ...and the modification's own row is its ALPHA CARBON's, token 3
    expect(rows[2][2]).toBe(3 * 100 + 3);
    expect(rows[2][0]).toBe(3 * 100 + 0);
  });

  it("refuses a matrix too narrow for the tokens it is asked for", () => {
    const values = new Float32Array(4 * 4);
    expect(() => matrixForViewer(values, [0, 1, 2, 9])).toThrow();
  });

  it("names the modified residues as viewer positions", () => {
    const batch = chainWithSep();
    const keep = viewerTokens(batch);
    // the phosphoserine is the third residue, so position 2
    expect(modifiedPositions(batch, keep)).toEqual([2]);
  });
});

describe("a real batch, featurised, against what the viewer will draw", () => {
  // A five-atom stand-in for a phosphoserine: a backbone, a side-chain atom
  // and a leaving OXT, which is what makes the atom count depend on where in
  // the chain it sits. The featuriser is the authority on the span; this test
  // exists because every assertion above rests on the SHAPE of one.
  const MODIFICATION = {
    code: "SEP",
    atoms: [
      { name: "N", element: 7, charge: 0, x: 0, y: 0, z: 0 },
      { name: "CA", element: 6, charge: 0, x: 1.5, y: 0, z: 0 },
      { name: "C", element: 6, charge: 0, x: 2.4, y: 0, z: 0 },
      { name: "O", element: 8, charge: 0, x: 3.0, y: 0, z: 0 },
      { name: "OG", element: 8, charge: 0, x: 1.5, y: 1.4, z: 0 },
      { name: "OXT", element: 8, charge: 0, x: 3.6, y: 0, z: 0, leaving: true },
    ],
    bonds: [{ from: 0, to: 1, order: 1 }, { from: 1, to: 2, order: 1 },
            { from: 2, to: 3, order: 2 }, { from: 1, to: 4, order: 1 }],
  };
  const SEQUENCE = "GWSTELEKHR";        // ten residues, the modification at 3

  const batchWith = (extra = {}) => featuriseProtein(SEQUENCE, {
    modifications: [{ chain: 0, position: 3, ...MODIFICATION }], ...extra });

  it("collapses exactly what the atomisation added", () => {
    const batch = batchWith();
    // five atoms mid-chain (the OXT leaves), so nine residues plus five tokens
    expect(batch.tokens).toBe(SEQUENCE.length + 4);
    expect(viewerTokens(batch).length).toBe(SEQUENCE.length);
  });

  it("keeps the alpha carbon, which is the atom the writer gives a backbone to", () => {
    const batch = batchWith();
    const span = batch.modifiedSpans[0];
    const kept = viewerTokens(batch).filter(
      (token) => token >= span.from && token < span.from + span.count);
    expect(kept.length).toBe(1);
    expect(span.atoms[kept[0] - span.from].name).toBe("CA");
  });

  it("writes the whole modification under ONE residue number, which is why it collapses", () => {
    const batch = batchWith();
    const positions = new Float32Array(batch.tokens * batch.dense * 3);
    const pdb = toPdb(batch, positions, undefined);
    const lines = pdb.split("\n").filter((line) => line.includes(" SEP "));
    expect(lines.length).toBe(batch.modifiedSpans[0].count);
    const numbers = new Set(lines.map((line) => line.slice(22, 26)));
    expect(numbers.size).toBe(1);
  });

  it("leaves a ligand's atoms as positions of their own", () => {
    const ligand = { code: "TST", atoms: [
      { name: "C1", element: 6, charge: 0, x: 0, y: 0, z: 0 },
      { name: "O1", element: 8, charge: 0, x: 1.4, y: 0, z: 0 }],
      bonds: [{ from: 0, to: 1, order: 1 }] };
    const batch = batchWith({ ligands: [ligand] });
    // nine ordinary residues + five modification tokens + two ligand atoms
    expect(viewerTokens(batch).length).toBe(SEQUENCE.length + ligand.atoms.length);
  });

  it("names the modified residue at the position the viewer gives it", () => {
    const batch = batchWith();
    expect(modifiedPositions(batch, viewerTokens(batch))).toEqual([2]);
  });
});
