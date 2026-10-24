import { describe, expect, it } from "./harness.js";
import { spanToContig, spansFromContig } from "../web/contig-ui.js";
import { RFD4P_PRESETS } from "../shared/rfd4p/presets.js";

describe("reading a contig back into spans", () => {
  // 🔴 THE FIELDS, NOT THE ROUND TRIP. This is the whole reason the bug below
  // survived a check against six shipped presets byte for byte: the editor read
  // `A16-35` as chain "A1", first 6, last 35 - and chain "A1" with first 6
  // joins back to exactly "A16-35". A round-trip test is blind to any mis-split
  // that re-joins to the same string, so what has to be compared is where the
  // chain ends and the number begins.
  it("splits the chain from the first residue", () => {
    expect(spansFromContig("A16-35")).toEqual([
      { kind: "segment", chain: "A", first: 16, last: 35 },
    ]);
    expect(spansFromContig("60-105,A16-35,A52-71")).toEqual([
      { kind: "generated", min: 60, max: 105 },
      { kind: "segment", chain: "A", first: 16, last: 35 },
      { kind: "segment", chain: "A", first: 52, last: 71 },
    ]);
  });

  // 🔴 A SEGMENT'S RANGE IS OPTIONAL. `parseContigToken` in
  // shared/rfd4p/pipeline/spec.js reads `A16` as one residue - `(?:-(\d+))?` -
  // and this editor refused it outright, so a preset using one fell through to
  // "this editor has no row for it yet" and could not be loaded at all.
  it("takes a single-residue segment", () => {
    expect(spansFromContig("A16")).toEqual([
      { kind: "segment", chain: "A", first: 16, last: 16 },
    ]);
  });

  // 🔴 AND THE CHAIN IS LETTERS ONLY, which is the invariant that makes the
  // mis-split impossible rather than merely absent from the cases above. A
  // chain id carrying a digit is a residue number that was eaten by the chain.
  it("never lets a digit into a chain id", () => {
    for (const preset of RFD4P_PRESETS) {
      const spans = spansFromContig(preset.contig);
      if (spans === null) continue;
      for (const span of spans) {
        if (span.kind !== "segment") continue;
        expect(/^[A-Za-z]+$/.test(span.chain)).toBe(true);
        expect(Number.isInteger(span.first)).toBe(true);
        expect(span.last >= span.first).toBe(true);
      }
    }
  });

  // The round trip is still worth asserting - it is just not sufficient. Every
  // shipped preset the editor claims to understand must come back byte for
  // byte, or loading one silently runs a different job from the one it names.
  it("round-trips every preset it accepts", () => {
    let accepted = 0;
    for (const preset of RFD4P_PRESETS) {
      const spans = spansFromContig(preset.contig);
      if (spans === null) continue;
      accepted += 1;
      const rebuilt = spans.map((span) => spanToContig(span)).filter((part) => part.length > 0);
      // Whitespace after a comma is cosmetic and some presets carry it
      // (`70-70, /0, A17-131`); what must survive is the tokens themselves.
      expect(rebuilt.join(",")).toBe(preset.contig.replace(/\s+/g, ""));
    }
    // 🔴 A GATE THAT STOPS MATCHING PASSES BY FINDING NOTHING. If a change made
    // `spansFromContig` return null everywhere, every assertion above would be
    // skipped and this file would still be green.
    expect(accepted >= 8).toBe(true);
  });
});
