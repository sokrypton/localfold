import { describe, expect, it } from "./harness.js";
import { METHOD_LIMITS, SPAN_KINDS, chainsOf, contigOf, emptySpan, spanToContig } from "../web/contig-ui.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";

const contig = (spans, method = "rfd4p") => contigOf(spans, method).contig;

describe("the design job as a list of spans", () => {
  // 🔴 THE EDITOR IS HELD TO RFD4-PROTEINA'S OWN PRESETS, which is the only
  // check that means anything: a contig the engine would not have written is a
  // contig the engine may not accept. These span lists are what a user builds
  // by adding rows, and the strings on the right are `shared/rfd4p/presets.js`
  // verbatim.
  const AS_SPANS = {
    uncond_short: [{ kind: "generated", min: 100, max: 100 }],
    ppi_pdl1: [
      { kind: "segment", chain: "A", first: 18, last: 132 },
      { kind: "break" },
      { kind: "generated", min: 76, max: 76 },
    ],
    motif_idx_1prw: [
      { kind: "generated", min: 5, max: 20 },
      { kind: "segment", chain: "A", first: 16, last: 35 },
      { kind: "generated", min: 10, max: 25 },
      { kind: "segment", chain: "A", first: 52, last: 71 },
      { kind: "generated", min: 5, max: 20 },
    ],
    sm_sam: [
      { kind: "generated", min: 110, max: 110 },
      { kind: "break" },
      { kind: "ligand", code: "SAM" },
    ],
    dna_7m5w: [
      { kind: "generated", min: 96, max: 96 },
      { kind: "break" },
      { kind: "segment", chain: "B", first: 1, last: 16 },
      { kind: "break" },
      { kind: "segment", chain: "C", first: 1, last: 16 },
    ],
  };

  for (const [id, spans] of Object.entries(AS_SPANS)) {
    it(`builds ${id}'s contig exactly`, () => {
      const preset = RFD4P_PRESETS.find((entry) => entry.id === id);
      expect(preset === undefined).toBe(false);
      expect(contig(spans)).toBe(preset.contig);
      expect(contigOf(spans).problems).toEqual([]);
    });
  }

  it("counts the tokens a job would have, as a range when spans have one", () => {
    const motif = contigOf(AS_SPANS.motif_idx_1prw);
    // 5-20 + 20 given + 10-25 + 20 given + 5-20
    expect(motif.tokens).toEqual({ min: 60, max: 105 });
    // ...and the unindexed preset reads 60-105 for the same protein, which is
    // the arithmetic agreeing with shared/rfd4p/design-spec.js reading the
    // finished structure back.
    expect(contigOf(AS_SPANS.ppi_pdl1).tokens).toEqual({ min: 191, max: 191 });
  });

  // 🔴 A ROW BEING FILLED IN EMITS NOTHING RATHER THAN SOMETHING PLAUSIBLE. A
  // contig assembled around a guess fails deep in the validation pipeline with
  // a message about token counts, which says nothing about the empty field that
  // caused it.
  it("refuses a span it cannot write yet, and says which", () => {
    expect(spanToContig({ kind: "ligand", code: "" })).toBe(null);
    expect(spanToContig({ kind: "ligand", code: "  " })).toBe(null);
    expect(spanToContig({ kind: "segment", chain: "", first: 1, last: 9 })).toBe(null);
    // backwards, which a number field lets you type
    expect(spanToContig({ kind: "segment", chain: "A", first: 50, last: 10 })).toBe(null);
    expect(spanToContig({ kind: "generated", min: 50, max: 10 })).toBe(null);
    expect(spanToContig({ kind: "generated", min: 0, max: 0 })).toBe(null);
    expect(contigOf([{ kind: "generated", min: 5, max: 1 }]).problems)
      .toContain("span 1 is not filled in yet");
  });

  it("refuses a list that asks for nothing", () => {
    expect(contigOf([]).problems).toContain("add at least one span");
    expect(contigOf([{ kind: "break" }, { kind: "break" }]).problems)
      .toContain("every span is a chain break, so nothing would be designed");
  });

  // 🔴 THE METHOD DECIDES WHICH KINDS ARE LEGAL, NOT WHAT A SPAN IS. Hunter
  // reaches a structure by folding a sequence, so it asks for ONE length; and it
  // has no pocket conditioning. Both are stated on the row rather than removed.
  it("marks what Protein Hunter cannot honour, and leaves RFD4 alone", () => {
    const range = [{ kind: "generated", min: 50, max: 120 }];
    expect(contigOf(range, "rfd4p").problems).toEqual([]);
    expect(contigOf(range, "hunter").problems).toContain(METHOD_LIMITS.hunter.range);
    // ...but one length is fine on both
    expect(contigOf([{ kind: "generated", min: 76, max: 76 }], "hunter").problems).toEqual([]);

    const ligand = [{ kind: "generated", min: 80, max: 80 }, { kind: "break" },
      { kind: "ligand", code: "HEM" }];
    expect(contigOf(ligand, "rfd4p").problems).toEqual([]);
    expect(contigOf(ligand, "hunter").problems).toContain(METHOD_LIMITS.hunter.ligand);
  });

  it("gives every kind a default a reader would expect", () => {
    for (const kind of SPAN_KINDS) {
      const span = emptySpan(kind);
      expect(span.kind).toBe(kind);
      // 🔴 THREE KINDS EMIT CONTIG NOTATION THE MOMENT THEY ARE ADDED. A
      // ligand waits for a code to be typed, and a GIVEN SEQUENCE never emits
      // any - a contig names coordinates, so Hunter reads that kind through
      // chainsOf instead. Both nulls are the design, not a gap.
      if (kind !== "ligand" && kind !== "sequence") {
        expect(typeof spanToContig(span)).toBe("string");
      } else {
        expect(spanToContig(span)).toBe(null);
      }
    }
  });

  // 🔴 ONE LIST, TWO DERIVATIONS, which is the whole reason this is a single
  // editor: `contigOf` writes RFD4-Proteina's notation and `chainsOf` writes
  // what Protein Hunter's loop takes. Neither method gets its own input screen.
  describe("the same spans as Protein Hunter's chains", () => {
    const TARGET = "GWSTELEKHREELKEFLKKEGITNVEIRIDN";

    it("reads a given chain and a designed one", () => {
      const spans = [{ kind: "sequence", value: TARGET }, { kind: "break" },
        { kind: "generated", min: 12, max: 12 }];
      expect(chainsOf(spans)).toEqual({
        chains: [TARGET, ""], chainIndex: 1, length: 12, problems: [],
      });
      // ...and the designed chain can come first, which is how v1 ordered them
      const first = chainsOf([{ kind: "generated", min: 20, max: 20 }, { kind: "break" },
        { kind: "sequence", value: TARGET }]);
      expect(first.chainIndex).toBe(0);
      expect(first.chains).toEqual(["", TARGET]);
    });

    it("refuses what Hunter cannot express", () => {
      // a motif inside a designed chain: one chain, both given and designed
      expect(chainsOf([{ kind: "sequence", value: "AAA" }, { kind: "generated", min: 5, max: 5 }])
        .problems).toContain("Protein Hunter cannot mix designed and given residues in one chain;"
          + " put a chain break between them");
      // two designed chains
      expect(chainsOf([{ kind: "generated", min: 5, max: 5 }, { kind: "break" },
        { kind: "generated", min: 6, max: 6 }]).problems)
        .toContain("Protein Hunter designs one chain at a time, not 2");
      // nothing designed at all
      expect(chainsOf([{ kind: "sequence", value: TARGET }]).problems)
        .toContain("nothing is being designed; add a Design span");
      expect(chainsOf([{ kind: "sequence", value: "" }, { kind: "break" },
        { kind: "generated", min: 9, max: 9 }]).problems).toContain("a given sequence is empty");
    });

    // 🔴 THE GEOMETRY AXIS, SHOWN AS A PAIR. A chain known only by its sequence
    // is exactly what Hunter takes and exactly what RFD4 cannot hold fixed,
    // because one reaches structure by folding and the other conditions on
    // coordinates. Same span, opposite verdicts.
    it("takes a sequence for Hunter and refuses it for RFD4", () => {
      const spans = [{ kind: "sequence", value: TARGET }, { kind: "break" },
        { kind: "generated", min: 12, max: 12 }];
      expect(chainsOf(spans).problems).toEqual([]);
      expect(contigOf(spans, "rfd4p").problems).toContain(METHOD_LIMITS.rfd4p.sequence);
      // ...and the structure-segment form is the opposite way round
      const segment = [{ kind: "segment", chain: "A", first: 18, last: 132 },
        { kind: "break" }, { kind: "generated", min: 76, max: 76 }];
      expect(contigOf(segment, "rfd4p").problems).toEqual([]);
    });

    // 🔴 A `From structure` SPAN REACHES HUNTER THROUGH ITS SEQUENCE, since
    // Hunter folds sequences and cannot be handed coordinates. Without a
    // structure loaded the span is REPORTED, not skipped: a hunt that quietly
    // dropped its target would fold the binder alone and score it against
    // nothing, which looks like a working run.
    it("reads a structure segment as a sequence, and refuses it without one", () => {
      const spans = [{ kind: "segment", chain: "A", first: 10, last: 14 },
        { kind: "break" }, { kind: "generated", min: 8, max: 8 }];
      expect(chainsOf(spans).problems).toContain("Protein Hunter folds sequences, so a"
        + " \u201cFrom structure\u201d span needs the structure loaded to read its sequence from");

      const sequenceOf = (chain, first, last) => (chain === "A" && first === 10 && last === 14
        ? "MKVLA" : null);
      expect(chainsOf(spans, { sequenceOf })).toEqual({
        chains: ["MKVLA", ""], chainIndex: 1, length: 8, problems: [],
      });

      // ...and a span naming residues the structure does not have says which
      const missing = [{ kind: "segment", chain: "Z", first: 1, last: 9 },
        { kind: "break" }, { kind: "generated", min: 8, max: 8 }];
      expect(chainsOf(missing, { sequenceOf }).problems)
        .toContain("chain Z 1-9 is not in the loaded structure");
    });

    it("ignores a ligand, which is context and not a chain", () => {
      const spans = [{ kind: "generated", min: 80, max: 80 }, { kind: "break" },
        { kind: "ligand", code: "HEM" }];
      expect(chainsOf(spans).chains).toEqual([""]);
      expect(chainsOf(spans).chainIndex).toBe(0);
    });
  });
});