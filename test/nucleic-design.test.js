import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "./harness.js";
import { decompress } from "../shared/rfd4p/vendor/fzstd.js";
import { baseLetter, designChain } from "../cpu/design/mpnn-bridge.js";
import { chooseDesigner, DESIGNERS } from "../cpu/design/designers.js";
import { designBias, sampleSequence } from "../cpu/design/sample-sequence.js";
import { contactOf, contactProblem } from "../cpu/design/contact.js";
import { Model } from "../cpu/design/mpnn/model.js";
import { NA_ALPHABET, NA_RESTYPE_TO_INT } from "../cpu/design/mpnn/na.js";
import { Weights } from "../cpu/design/mpnn/weights.js";
import { chainsOf } from "../web/contig-ui.js";
import { runDesign } from "../webgpu/design/hunter-loop.js";
import { uniformFrom } from "../webgpu/af3/fold.js";

// 🔴 PROTEIN HUNTER DESIGNING A DNA OR RNA CHAIN (2026-10-28), with NA-MPNN - piece by piece, and
// on 7M5W (capicua's HMG domain, chain A, on a duplex B:C) where the right answer is known: a strand
// designed against its partner should come back as the partner's reverse complement.

const path = (name) => fileURLToPath(new URL(name, import.meta.url));
const cif = new TextDecoder().decode(decompress(new Uint8Array(readFileSync(path("../tools/fixtures/rcsb/7m5w.cif.zst")))));
const naMpnn = () => {
  const bytes = readFileSync(path(`../web/public/mpnn/${DESIGNERS.na.file}`));
  return new Model(Weights.fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
};
const reverseComplement = (strand) => [...strand].reverse()
  .map((base) => ({ A: "T", T: "A", G: "C", C: "G" })[base]).join("");

describe("designing a nucleic chain", () => {
  it("is NA-MPNN's job alone", () => {
    expect(chooseDesigner({ designs: "dna" }).name).toBe("na");
    expect(chooseDesigner({ designs: "rna", ligands: 1 }).name).toBe("na");
    expect(chooseDesigner({ designs: "rna", ligands: 1 }).why).toContain("does not see the ligand");
    // ...and a protein design is unchanged
    expect(chooseDesigner({ nucleic: 2 }).name).toBe("na");
    expect(chooseDesigner({ ligands: 1 }).name).toBe("ligand");
  });

  it("starts mostly N, the rest drawn from the chain's own bases", () => {
    const dna = sampleSequence(40, { kind: "dna", percentX: 50, random: uniformFrom(3) });
    expect(/^[ACGTN]+$/.test(dna)).toBe(true);
    expect([...dna].filter((base) => base === "N").length).toBe(20);
    expect(/^[ACGUN]+$/.test(sampleSequence(40, { kind: "rna", percentX: 50, random: uniformFrom(3) }))).toBe(true);
  });

  it("samples only the four DNA tokens, which shared tokens use for RNA too", () => {
    const bias = designBias(2, { alphabet: NA_ALPHABET, designs: "nucleic", omit: "C" });
    const open = [...NA_ALPHABET].filter((_, at) => bias[at] === 0).join("");
    expect(open).toBe("acgt");
    expect(bias[NA_ALPHABET.length + NA_ALPHABET.indexOf("g")]).toBe(0);
    // a 21-letter alphabet has no nucleotide to allow
    expect(() => designBias(2, { designs: "nucleic" })).toThrow(/only NA-MPNN/);
  });

  it("reads a token back as the letter the fold takes", () => {
    expect(baseLetter(NA_RESTYPE_TO_INT.DT, "dna")).toBe("T");
    expect(baseLetter(NA_RESTYPE_TO_INT.DT, "rna")).toBe("U");
    expect(baseLetter(NA_RESTYPE_TO_INT.DX, "dna")).toBe("N");
    expect(() => baseLetter(NA_RESTYPE_TO_INT.LYS, "dna")).toThrow(/put LYS in a DNA chain/);
  });

  it("designs 7M5W's strand C as strand B's reverse complement", () => {
    const model = naMpnn();
    const result = designChain(model, { pdb: cif, chain: "C", temperature: 0.1, random: uniformFrom(1), kind: "dna" });
    const [protein, strandB] = result.full.split(":");
    const partner = reverseComplement(strandB.toUpperCase());
    expect(result.sequence).toHaveLength(18);
    expect(/^[ACGT]+$/.test(result.sequence)).toBe(true);
    // measured: GCTTTTTCATTCATAACT, 17 of 18, at three seeds - the bound leaves room for a seed
    const same = [...result.sequence].filter((base, at) => base === partner[at]).length;
    expect(same >= 15).toBe(true);
    expect(protein.length > 100).toBe(true);
    // ...and a protein designer is refused rather than asked
    const bytes = readFileSync(path(`../web/public/mpnn/${DESIGNERS.soluble.file}`));
    const soluble = new Model(Weights.fromArrayBuffer(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
    expect(() => designChain(soluble, { pdb: cif, chain: "C", kind: "dna" })).toThrow(/only NA-MPNN designs a DNA chain/);
  });
});

describe("a designed nucleic chain in the job and the loop", () => {
  it("keeps the Designed segment's DNA, starting from N", () => {
    const job = chainsOf([
      { kind: "sequence", value: "GTTATGAAT", polymer: "dna" }, { kind: "break" },
      { kind: "generated", min: 10, max: 10, polymer: "dna" },
    ]);
    expect(job.problems).toEqual([]);
    expect(job.chainKinds).toEqual(["dna", "dna"]);
    expect(job.chainIndex).toBe(1);
    expect(job.chains[1]).toBe("");
    // a part-designed strand arrives shaped, its designed part as N
    const mixed = chainsOf([{ kind: "sequence", value: "ACGT", polymer: "dna", redesign: false },
      { kind: "generated", min: 3, max: 3, polymer: "dna" }]);
    expect(mixed.chains[0]).toBe("ACGTNNN");
    expect(chainsOf([{ kind: "sequence", value: "ACGT", polymer: "dna" },
      { kind: "generated", min: 3, max: 3, polymer: "protein" }]).problems.join(" ")).toContain("part protein and part nucleic");
  });

  it("has no alanine ceiling for a strand of adenines", async () => {
    const records = [];
    for await (const record of runDesign({
      chains: ["", "ACGT"], chainIndex: 0, length: 5, percentX: 100, seed: 1, cycles: 1, kind: "dna",
      fold: async () => ({ pdb: "PDB", confidence: { meanPlddt: 80, iptm: 0.7, chainPairIptm: { "1|2": 0.7 } } }),
      design: async (_pdb, context) => ({ sequence: context.kind === "dna" ? "AAAAA" : "XXXXX", score: -1 }),
    })) records.push(record);
    expect(records[0].sequence).toBe("NNNNN");
    expect(records[1].sequence).toBe("AAAAA");
    expect(records[1].alanine).toBe(null);
    expect(records[1].best).toBe(true);
  });
});

describe("a strand of unknown bases", () => {
  // 🔴 AF3 WRITES AN UNKNOWN BASE AS "DN" (RNA's "N"), and a hunt's first cycle folds a strand of
  // them. NA-MPNN's parser did not know the name, so it kept only the strand's real bases and the
  // design came back one letter long - measured on the page, cycle after cycle of "G".
  it("is read as the whole strand, each an unknown NA-MPNN position", () => {
    const columns = cif.split("\n").filter((line) => line.startsWith("_atom_site.")).map((line) => line.trim());
    const chain = columns.indexOf("_atom_site.auth_asym_id");
    const names = ["_atom_site.label_comp_id", "_atom_site.auth_comp_id"].map((name) => columns.indexOf(name));
    const unknown = cif.split("\n").map((line) => {
      if (!/^(ATOM|HETATM)/.test(line)) return line;
      const fields = line.trim().split(/\s+/);
      if (fields[chain] !== "C" || !/^D[ACGT]$/.test(fields[names[0]])) return line;
      for (const at of names) fields[at] = "DN";
      return fields.join(" ");
    }).join("\n");
    const result = designChain(naMpnn(), { pdb: unknown, chain: "C", temperature: 0.1, random: uniformFrom(1), kind: "dna" });
    // ...the crystal's 5' base has no phosphate, so NA-MPNN cannot read it and it stays N
    expect(result.sequence).toHaveLength(18);
    expect(/^N[ACGT]{17}$/.test(result.sequence)).toBe(true);
  });
});

describe("the contact check sees DNA", () => {
  it("measures a designed strand against its partner and the protein", () => {
    const strand = contactOf(cif, { chain: "C" });
    expect(strand.chains).toEqual(["A", "B", "C"]);
    expect(strand.contacts > 0).toBe(true);
    expect(contactProblem(cif, { chain: "C" })).toBe(null);
    // a protein binder against DNA alone used to have nothing to touch and so no opinion
    const binder = contactOf(cif, { chain: "A" });
    expect(binder.touched.some((key) => key.startsWith("B:") || key.startsWith("C:"))).toBe(true);
  });
});
