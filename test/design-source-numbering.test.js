import { describe, expect, it } from "./harness.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { sourceResidues, translateSpans } from "../cpu/design/source-numbering.js";
import { parseCifToAtomArray } from "../shared/rfd4p/pipeline/cif.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";
import { spanToContig, spansFromContig } from "../web/contig-ui.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";

const example = (name) => {
  const path = fileURLToPath(new URL(`../web/examples/${name}`, import.meta.url));
  const bytes = new Uint8Array(readFileSync(path));
  return parseCifToAtomArray(new TextDecoder().decode(
    name.endsWith(".zst") ? decompress(bytes) : bytes));
};

describe("a cropped example's contig, in the crop's own numbering", () => {
  // 🔴 THE TWO NUMBERINGS DISAGREE AND BOTH LOOK LIKE REAL RESIDUE IDS.
  // `motif_idx_1prw` carries chain A at 0-42 and its contig asks for A16-35 and
  // A52-71 - the deposited entry's numbering. Read literally, half that contig
  // names residues the file does not contain and the other half names the WRONG
  // ones, which is worse, because it resolves.
  it("moves a motif preset onto the structure it ships with", () => {
    const atoms = example("motif_idx_1prw.cif.zst");
    const preset = RFD4P_PRESETS.find((entry) => entry.id === "motif_idx_1prw");
    const { spans, translated } = translateSpans(spansFromContig(preset.contig), atoms);
    expect(translated).toBe(2);
    expect(spans.map(spanToContig).filter((part) => part.length > 0).join(","))
      .toBe("5-20,A1-20,10-25,A22-41,5-20");
  });

  // The provenance is a per-atom annotation, so a file that has none is its own
  // source and must come back untouched - every ordinary PDB entry a user loads.
  it("is a no-op on a structure with no provenance", () => {
    const atoms = example("uncond_short.cif.zst");
    expect(sourceResidues(atoms).size).toBe(0);
    const spans = [{ kind: "segment", chain: "A", first: 3, last: 9 }];
    const { spans: out, translated } = translateSpans(spans, atoms);
    expect(translated).toBe(0);
    expect(out).toBe(spans);
  });

  // 🔴 A RANGE THAT CANNOT BE EXPRESSED AS ONE RANGE IS LEFT ALONE, so the
  // page's own "is not in the loaded structure" still fires and says the true
  // thing. `ppi_pdl1` is exactly this: its crop records A17-A131 and its contig
  // asks A18-132 - the same count, offset by one - so the last residue has no
  // source and there is no honest range to produce. Guessing one would turn a
  // visible refusal into a silent wrong answer.
  it("refuses to guess when a residue has no source", () => {
    const atoms = example("ppi_pdl1.cif.zst");
    expect(sourceResidues(atoms).size).toBe(115);
    const spans = [{ kind: "segment", chain: "A", first: 18, last: 132 }];
    const { translated } = translateSpans(spans, atoms);
    expect(translated).toBe(0);
  });

  // ...and a range wholly inside the provenance does translate, so the refusal
  // above is about the missing residue and not about the file.
  it("translates a range that is wholly covered", () => {
    const atoms = example("ppi_pdl1.cif.zst");
    const { spans, translated } = translateSpans(
      [{ kind: "segment", chain: "A", first: 18, last: 131 }], atoms);
    expect(translated).toBe(1);
    expect(spans[0]).toEqual({ kind: "segment", chain: "B", first: 2, last: 115 });
  });

  // 🔴 CONTIGUITY IS CHECKED. A contig range is one SEGMENT; if its residues
  // landed out of order or on two chains there is no single range that means
  // the same thing, and returning the ends would include whatever fell between.
  it("refuses a range whose residues are not contiguous in the crop", () => {
    const atoms = {
      nAtoms: 3,
      chain_id: ["A", "A", "A"],
      res_id: [1, 5, 6],
      annotations: { "annotation_src-component_1_atom": ["A10", "A11", "A12"] },
    };
    expect(translateSpans([{ kind: "segment", chain: "A", first: 10, last: 12 }],
      atoms).translated).toBe(0);
    expect(translateSpans([{ kind: "segment", chain: "A", first: 11, last: 12 }],
      atoms).translated).toBe(1);
  });
});
