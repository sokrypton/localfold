import { describe, expect, it } from "./harness.js";
import { METHOD_LIMITS, SPAN_KINDS, chainsOf, conditionsOf, contigOf, emptySpan, hotspotsOf,
  modificationsOf, spanToContig } from "../web/contig-ui.js";
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
    // 🔴 A RANGE IS NO LONGER ONE OF THEM. It used to be refused - "Hunter
    // folds one length at a time; a range needs a search over lengths" - which
    // is true of a single fold and describes a search the page already has:
    // `Runs` repeats the hunt, and each run draws its own length out of the
    // range (see test/design-length-range.test.js). Both methods take one now.
    const range = [{ kind: "generated", min: 50, max: 120 }];
    expect(contigOf(range, "rfd4p").problems).toEqual([]);
    expect(contigOf(range, "hunter").problems).toEqual([]);
    expect(METHOD_LIMITS.hunter.range).toBe(undefined);
    // ...and one length is still fine on both
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
      // 🔴 THE FIELDS THIS TEST IS ABOUT, not the whole object: a deep-equal
      // here broke on every field later added to the return - polymer kinds,
      // ligands, held positions, modifications - each time reporting a passing
      // feature as a failure.
      const read = chainsOf(spans);
      expect(read.chains).toEqual([TARGET, ""]);
      expect(read.chainIndex).toBe(1);
      expect(read.length).toBe(12);
      expect(read.problems).toEqual([]);
      // ...and the designed chain can come first, which is how v1 ordered them
      const first = chainsOf([{ kind: "generated", min: 20, max: 20 }, { kind: "break" },
        { kind: "sequence", value: TARGET }]);
      expect(first.chainIndex).toBe(0);
      expect(first.chains).toEqual(["", TARGET]);
    });

    it("refuses what Hunter cannot express", () => {
      // two designed chains
      expect(chainsOf([{ kind: "generated", min: 5, max: 5 }, { kind: "break" },
        { kind: "generated", min: 6, max: 6 }]).problems)
        .toContain("Protein Hunter designs one chain at a time");
      // nothing designed at all
      expect(chainsOf([{ kind: "sequence", value: TARGET }]).problems)
        .toContain("nothing is being designed; add a Design span or mark a given one Redesign");
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
        + " \u201cKeep from structure\u201d span needs the structure loaded to read its sequence from");

      const sequenceOf = (chain, first, last) => (chain === "A" && first === 10 && last === 14
        ? "MKVLA" : null);
      const read = chainsOf(spans, { sequenceOf });
      expect(read.chains).toEqual(["MKVLA", ""]);
      expect(read.chainIndex).toBe(1);
      expect(read.length).toBe(8);
      expect(read.problems).toEqual([]);

      // ...and a span naming residues the structure does not have says which
      const missing = [{ kind: "segment", chain: "Z", first: 1, last: 9 },
        { kind: "break" }, { kind: "generated", min: 8, max: 8 }];
      expect(chainsOf(missing, { sequenceOf }).problems)
        .toContain("chain Z 1-9 is not in the loaded structure");
    });

    // 🔴 WHAT A CHAIN IS MADE OF CHANGES THE FOLD, NOT ONLY THE DESIGNER. The
    // featuriser reads a nucleic chain one token per base and gives each ligand
    // a chain of its own, so a DNA target folded as protein is not a slower
    // answer - it is a different one. The kinds travel with the chains.
    it("carries each chain's polymer and the ligands beside them", () => {
      const dna = chainsOf([{ kind: "sequence", value: "ATGCATGC", polymer: "dna" },
        { kind: "break" }, { kind: "generated", min: 20, max: 20 }]);
      expect(dna.chainKinds).toEqual(["dna", "protein"]);
      expect(dna.chains).toEqual(["ATGCATGC", ""]);

      const withLigand = chainsOf([{ kind: "generated", min: 60, max: 60 },
        { kind: "break" }, { kind: "ligand", code: "hem" }]);
      expect(withLigand.ligandCodes).toEqual(["HEM"]);
      // a designed chain is protein whatever else is in the job
      expect(withLigand.chainKinds).toEqual(["protein"]);

      // 🔴 A CHAIN IS ONE POLYMER. Mixing two inside one chain is reported
      // rather than resolved, because either resolution is a different molecule
      // from the one asked for.
      expect(chainsOf([{ kind: "sequence", value: "AAA" },
        { kind: "sequence", value: "ATGC", polymer: "dna" },
        { kind: "break" }, { kind: "generated", min: 9, max: 9 }]).problems)
        .toContain("a chain cannot be part protein and part nucleic;"
          + " put a chain break between them");
    });

    // 🔴 ONE CHAIN MIXING GIVEN AND DESIGNED RESIDUES is the primitive most of
    // the remaining modalities need, and Hunter could not express it at all
    // before: it designed whole chains only. The held positions are the given
    // parts' indices inside the chain, which is exactly what designChain zeroes
    // in the chain mask.
    it("scaffolds a motif: X around held residues, in one chain", () => {
      const sequenceOf = (chain, first, last) => (chain === "A"
        ? "MKVLATGGAWRECKLMNPQ".slice(0, last - first + 1) : null);
      const motif = chainsOf([{ kind: "generated", min: 5, max: 5 },
        { kind: "segment", chain: "A", first: 1, last: 8 },
        { kind: "generated", min: 5, max: 5 }], { sequenceOf });
      // 🔴 PRE-SHAPED, because the loop cannot know where the motif goes - unlike
      // a wholly designed chain, which is handed over EMPTY so the loop draws
      // its own opening sequence (the percent-X ramp is the method, not a
      // detail). The ramp therefore does not apply to a scaffolded chain, which
      // is a deviation in a case the reference cannot express at all.
      expect(motif.chains).toEqual(["XXXXXMKVLATGGXXXXX"]);
      expect(motif.hold).toEqual([5, 6, 7, 8, 9, 10, 11, 12]);
      expect(motif.chainIndex).toBe(0);
      expect(motif.problems).toEqual([]);
    });

    // 🔴 `redesign` IS THE SEQUENCE AXIS MADE EDITABLE, and these two cases are
    // the same flag used at two scales: the whole chain is inverse folding, part
    // of one is an antibody's CDR.
    it("frees a given span when it is marked redesign", () => {
      const sequenceOf = (chain, first, last) => (chain === "A"
        ? "MKVLATGGAWRECKLMNPQ".slice(0, last - first + 1) : null);
      const inverse = chainsOf([{ kind: "segment", chain: "A", first: 1, last: 10,
        redesign: true }], { sequenceOf });
      expect(inverse.chains).toEqual(["MKVLATGGAW"]);
      expect(inverse.hold).toEqual([]);
      expect(inverse.chainIndex).toBe(0);

      const cdr = chainsOf([
        { kind: "segment", chain: "A", first: 1, last: 5 },
        { kind: "segment", chain: "A", first: 6, last: 9, redesign: true },
        { kind: "segment", chain: "A", first: 10, last: 14 },
      ], { sequenceOf });
      expect(cdr.hold).toEqual([0, 1, 2, 3, 4, 9, 10, 11, 12, 13]);
      expect(cdr.problems).toEqual([]);
    });

    it("refuses a designed chain with every residue held", () => {
      const sequenceOf = () => "MKVLA";
      expect(chainsOf([{ kind: "segment", chain: "A", first: 1, last: 5 },
        { kind: "generated", min: 0, max: 0 }], { sequenceOf }).problems)
        .toContain("every residue of the designed chain is held,"
          + " so there is nothing to design");
    });

    // 🔴 THE LOOP'S OWN OPENING MOVE IS PRESERVED for the ordinary case: a
    // wholly designed chain is handed over EMPTY, which is what makes the loop
    // draw its mostly-X starting sequence rather than being given a constant.
    it("hands a wholly designed chain over empty, so the loop draws it", () => {
      const plain = chainsOf([{ kind: "sequence", value: "MKVLA" }, { kind: "break" },
        { kind: "generated", min: 12, max: 12 }]);
      expect(plain.chains).toEqual(["MKVLA", ""]);
      expect(plain.length).toBe(12);
      expect(plain.hold).toEqual([]);
    });

    // 🔴 A MODIFIED RESIDUE IS A PTM TARGET, which v1 could not express at all.
    // The position is 1-based WITHIN THE SPAN because that is how a person reads
    // a paper, and chainsOf resolves it to the chain letter and offset the fold
    // uses.
    it("places modified residues on the right chain and position", () => {
      const spans = [{ kind: "sequence", value: "GWSTELEKHREELK", modifications: "3:SEP 7:TPO" },
        { kind: "break" }, { kind: "generated", min: 12, max: 12 }];
      expect(modificationsOf(spans)).toEqual([
        { chainOffset: 0, position: 3, code: "SEP" },
        { chainOffset: 0, position: 7, code: "TPO" },
      ]);
      expect(chainsOf(spans).modifications).toEqual([
        { chain: "A", position: 3, code: "SEP" },
        { chain: "A", position: 7, code: "TPO" },
      ]);

      // ...offset past earlier spans of the same chain, and past a break
      const later = [{ kind: "sequence", value: "AAAAA" },
        { kind: "sequence", value: "GGGGG", modifications: "2:SEP" },
        { kind: "break" },
        { kind: "sequence", value: "CCCCC", modifications: "1:TPO" },
        { kind: "break" }, { kind: "generated", min: 6, max: 6 }];
      expect(chainsOf(later).modifications).toEqual([
        { chain: "A", position: 7, code: "SEP" },
        { chain: "B", position: 1, code: "TPO" },
      ]);

      // 🔴 AND A POSITION PAST THE CHAIN'S END IS REPORTED. The fold would
      // otherwise fetch a CCD component and place it nowhere.
      expect(chainsOf([{ kind: "sequence", value: "AAA", modifications: "9:SEP" },
        { kind: "break" }, { kind: "generated", min: 5, max: 5 }]).problems)
        .toContain("chain A has 3 residues, so position 9 cannot be modified");
    });

    it("ignores a ligand, which is context and not a chain", () => {
      const spans = [{ kind: "generated", min: 80, max: 80 }, { kind: "break" },
        { kind: "ligand", code: "HEM" }];
      expect(chainsOf(spans).chains).toEqual([""]);
      expect(chainsOf(spans).chainIndex).toBe(0);
    });
  });

  // 🔴 HOTSPOTS ARE THE ONE CONDITION A USER CAN REASONABLY NAME, and the
  // bracketed residue list is not cosmetic: evalPathSelection splits on commas
  // before slashes, so a bare list conditions on its first residue alone and
  // says nothing about it. Two shipped presets were doing exactly that - see
  // test/rfd4p-selection.test.js.
  // 🔴 A LABELLED SPAN IS THE HANDLE A CONDITION SELECTS A STRETCH BY, and this
  // is the last thing the editor could not express: `sel('*/h1/*/*')` reads
  // field two of the path, the res_name, so a span's label becomes the name of
  // residues that do not exist yet. Held to the shipped fold-conditioned preset,
  // contig AND conditions.
  describe("labelled spans and the structure they ask for", () => {
    const ROSSMANN = [
      { kind: "generated", min: 16, max: 16, name: "h1", secondary: "HELIX" },
      { kind: "generated", min: 4, max: 4, name: "l1", secondary: "LOOP" },
      { kind: "generated", min: 8, max: 8, name: "s1", secondary: "SHEET" },
      { kind: "generated", min: 4, max: 4, name: "l2", secondary: "LOOP" },
      { kind: "generated", min: 14, max: 14, name: "h2", secondary: "HELIX" },
      { kind: "generated", min: 8, max: 8, name: "s2", secondary: "SHEET" },
    ];

    it("builds custom_cath_monomer's contig and its C_SEC exactly", () => {
      const preset = RFD4P_PRESETS.find((entry) => entry.id === "custom_cath_monomer");
      expect(contigOf(ROSSMANN, "rfd4p").contig).toBe(preset.contig);
      expect(conditionsOf(ROSSMANN).C_SEC).toEqual(preset.conditions.C_SEC);
      expect(contigOf(ROSSMANN, "rfd4p").problems).toEqual([]);
    });

    it("suffixes the polymer, and refuses a name that is not a name", () => {
      expect(spanToContig({ kind: "generated", min: 6, max: 6, name: "r1", polymer: "rna" }))
        .toBe("r1:6-6R");
      expect(spanToContig({ kind: "generated", min: 6, max: 6, name: "d1", polymer: "dna" }))
        .toBe("d1:6-6D");
      // an unnamed span is the plain form, unchanged
      expect(spanToContig({ kind: "generated", min: 6, max: 6 })).toBe("6-6");
      // ...and a label with a space or a leading digit is not a label
      expect(spanToContig({ kind: "generated", min: 6, max: 6, name: "1h" })).toBe(null);
      expect(spanToContig({ kind: "generated", min: 6, max: 6, name: "h 1" })).toBe(null);
    });

    // 🔴 AND HUNTER CANNOT BE ASKED. It reaches a structure by folding a
    // sequence, so there is no way to tell it "this stretch should be a helix";
    // a named span there would be a control that did nothing.
    it("marks a labelled span unavailable on Protein Hunter", () => {
      expect(contigOf(ROSSMANN, "hunter").problems).toContain(METHOD_LIMITS.hunter.labelled);
      // an unnamed span of the same lengths is fine
      expect(contigOf([{ kind: "generated", min: 16, max: 16 }], "hunter").problems).toEqual([]);
    });
  });

  describe("the conditions a span list asks for", () => {
    it("writes hotspots as a bracketed selection, per chain", () => {
      expect(conditionsOf([{ kind: "segment", chain: "A", first: 18, last: 132,
        hotspots: "54 56 66 115" }]))
        .toEqual({ C_HOT: { true: [{ select: "sel('A/*/[54,56,66,115]/*')" }] } });
      // separators are whatever the user typed
      expect(conditionsOf([{ kind: "segment", chain: "A", first: 1, last: 9, hotspots: "10, 12" }]))
        .toEqual({ C_HOT: { true: [{ select: "sel('A/*/[10,12]/*')" }] } });
      // one selection per chain
      const two = conditionsOf([
        { kind: "segment", chain: "A", first: 1, last: 9, hotspots: "10" },
        { kind: "segment", chain: "B", first: 1, last: 9, hotspots: "3" },
      ]);
      expect(two.C_HOT.true.length).toBe(2);
    });

    it("asks for nothing when no hotspot is named", () => {
      expect(conditionsOf([{ kind: "generated", min: 10, max: 10 }])).toBe(undefined);
      expect(conditionsOf([{ kind: "segment", chain: "A", first: 1, last: 9, hotspots: "" }]))
        .toBe(undefined);
      expect(conditionsOf([])).toBe(undefined);
    });

    // 🔴 ONE FIELD, TWO MECHANISMS, AND BOTH METHODS CAN NOW HONOUR IT. RFD4
    // turns hotspots into a C_HOT condition and generates against them; Protein
    // Hunter has no conditioning at all, so it folds and then REJECTS a cycle
    // whose binder is not touching them (cpu/design/contact.js). The same thing
    // asked for, reached two different ways - which is why the editor does not
    // need a per-method screen.
    it("offers hotspots to both methods", () => {
      const spans = [{ kind: "segment", chain: "A", first: 1, last: 40, hotspots: "12 14" },
        { kind: "break" }, { kind: "generated", min: 30, max: 30 }];
      expect(contigOf(spans, "hunter").problems).toEqual([]);
      expect(contigOf(spans, "rfd4p").problems).toEqual([]);
      // RFD4 reads them as a condition...
      expect(conditionsOf(spans)).toEqual({
        C_HOT: { true: [{ select: "sel('A/*/[12,14]/*')" }] },
      });
      // ...and Hunter as plain numbers to measure against
      expect(hotspotsOf(spans)).toEqual({ A: [12, 14] });
      expect(hotspotsOf([{ kind: "generated", min: 9, max: 9 }])).toBe(undefined);
    });
  });
});