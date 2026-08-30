import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "./harness.js";
import { FALLBACK_BANDS, hydropathyColor, KYTE_DOOLITTLE,
  RESIDUES_BY_HYDROPATHY } from "../web/hydrophobicity.js";

// The page reads py2Dmol's bands at runtime, so the copy in web/hydrophobicity.js
// only stands in before the vendor script has run. It still has to agree: a
// legend that disagrees with the picture is worse than no legend. Checked
// against the sibling checkout when there is one, skipped when there is not,
// because a test that needs a neighbouring repository is not a test everyone
// can run.
const PY2DMOL = "../py2Dmol/src/core/mol.js";
const available = existsSync(PY2DMOL);

/** py2Dmol's own band table, parsed one line at a time. */
function upstreamBands() {
  const source = readFileSync(PY2DMOL, "utf8");
  const block = source.match(/const HYDROPHOBICITY_BANDS = \[([\s\S]*?)\n\];/)[1];
  return block.split("\n")
    .map((line) => /min:\s*(-?[\w.]+),\s*hex:\s*'(#[0-9a-f]{6})',\s*label:\s*'([^']+)'/.exec(line))
    .filter((match) => match !== null)
    .map((match) => ({ min: Number(match[1]), hex: match[2], label: match[3] }));
}

describe("hydropathy bands", () => {
  it.skipIf(!available)("match the ones py2Dmol draws with", () => {
    const theirs = upstreamBands();
    expect(theirs.length).toBe(FALLBACK_BANDS.length);
    expect(theirs).toEqual(FALLBACK_BANDS);
  });

  it("covers all twenty amino acids, plus X", () => {
    expect(Object.keys(KYTE_DOOLITTLE).length).toBe(21);
    for (const code of "ARNDCQEGHILKMFPSTWYV") {
      expect(typeof KYTE_DOOLITTLE[code]).toBe("number");
    }
  });

  it("puts each residue in the band its hydropathy asks for", () => {
    // the extremes of the scale, and the boundaries between buckets
    expect(hydropathyColor("I")).toBe("#f2994a");   // 4.5, very hydrophobic
    expect(hydropathyColor("L")).toBe("#f2994a");   // 3.8
    expect(hydropathyColor("F")).toBe("#f2c94c");   // 2.8, hydrophobic
    expect(hydropathyColor("A")).toBe("#f2c94c");   // 1.8
    expect(hydropathyColor("G")).toBe("#cfd8d4");   // -0.4, neutral
    expect(hydropathyColor("W")).toBe("#cfd8d4");   // -0.9
    expect(hydropathyColor("Y")).toBe("#56b9dc");   // -1.3, hydrophilic
    expect(hydropathyColor("P")).toBe("#56b9dc");   // -1.6
    // ...histidine is -3.2, which is BELOW the -3.0 cutoff, so it is very
    // hydrophilic rather than hydrophilic. The bands are `value >= min`.
    expect(hydropathyColor("H")).toBe("#187bd1");
    expect(hydropathyColor("K")).toBe("#187bd1");   // -3.9, very hydrophilic
    expect(hydropathyColor("R")).toBe("#187bd1");   // -4.5
  });

  it("orders the twenty most hydrophobic first, with no gaps", () => {
    expect(RESIDUES_BY_HYDROPATHY.length).toBe(20);
    expect([...RESIDUES_BY_HYDROPATHY].sort().join("")).toBe("ACDEFGHIKLMNPQRSTVWY");
    expect(RESIDUES_BY_HYDROPATHY[0]).toBe("I");                       // 4.5
    expect(RESIDUES_BY_HYDROPATHY[19]).toBe("R");                      // -4.5
    for (let i = 1; i < RESIDUES_BY_HYDROPATHY.length; i += 1) {
      const previous = KYTE_DOOLITTLE[RESIDUES_BY_HYDROPATHY[i - 1]];
      expect(KYTE_DOOLITTLE[RESIDUES_BY_HYDROPATHY[i]] <= previous).toBe(true);
    }
  });

  it("lays the colours out as five unbroken runs, so the row IS the scale", () => {
    // ...the whole reason for sorting: a colour must not reappear after another
    const colours = RESIDUES_BY_HYDROPATHY.map(hydropathyColor);
    const runs = colours.filter((colour, i) => colour !== colours[i - 1]);
    expect(runs.length).toBe(5);
    expect(new Set(runs).size).toBe(5);
    expect(runs).toEqual(["#f2994a", "#f2c94c", "#cfd8d4", "#56b9dc", "#187bd1"]);
  });

  it("treats an unknown residue as neutral rather than throwing", () => {
    expect(hydropathyColor("X")).toBe("#cfd8d4");
    expect(hydropathyColor("?")).toBe("#cfd8d4");
  });
});
