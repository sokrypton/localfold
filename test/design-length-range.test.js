import { describe, expect, it } from "./harness.js";
import { METHOD_LIMITS, chainsOf, contigOf, drawLengths } from "../web/contig-ui.js";

const ranged = [{ kind: "generated", min: 5, max: 20 },
  { kind: "segment", chain: "A", first: 16, last: 35 },
  { kind: "generated", min: 10, max: 25 }];

describe("a length range is a search over lengths", () => {
  // 🔴 EVERY DRAW IS INSIDE THE RANGE. A contig saying 5-20 that folds 4 or 21
  // is designing something nobody asked for, and the fold would succeed.
  it("draws inside the range", () => {
    for (let run = 0; run < 64; run += 1) {
      for (const span of drawLengths(ranged, { run, seed: 7 })) {
        if (span.kind !== "generated") continue;
        expect(span.min).toBe(span.max);
      }
      const [first, , second] = drawLengths(ranged, { run, seed: 7 });
      expect(first.min >= 5 && first.min <= 20).toBe(true);
      expect(second.min >= 10 && second.min <= 25).toBe(true);
    }
  });

  // 🔴 THE SAME SEED IS THE SAME HUNT. A draw from Math.random would make a run
  // impossible to reproduce or to report, which is the whole reason this is a
  // hash rather than a sample.
  it("is deterministic in the seed and the run", () => {
    const once = drawLengths(ranged, { run: 3, seed: 11 });
    const twice = drawLengths(ranged, { run: 3, seed: 11 });
    expect(once).toEqual(twice);
    const elsewhere = drawLengths(ranged, { run: 4, seed: 11 });
    expect(JSON.stringify(once) === JSON.stringify(elsewhere)).toBe(false);
  });

  // 🔴 AND THE RUNS MUST ACTUALLY DIFFER, or `Runs` is not a search at all and
  // the range is honoured in the notation while every run folds one size.
  it("covers more than one length across runs", () => {
    const seen = new Set();
    for (let run = 0; run < 16; run += 1) seen.add(drawLengths(ranged, { run, seed: 0 })[0].min);
    expect(seen.size > 1).toBe(true);
  });

  // Two ranged spans are independent choices; spreading them together would
  // only ever try "all short" then "all long".
  it("decorrelates two spans", () => {
    const pairs = new Set();
    for (let run = 0; run < 32; run += 1) {
      const drawn = drawLengths(ranged, { run, seed: 0 });
      pairs.add(`${drawn[0].min}:${drawn[2].min}`);
    }
    expect(pairs.size > 4).toBe(true);
  });

  // A job with nothing ranged must be untouched, so every existing hunt is
  // byte-for-byte the hunt it was.
  it("leaves a fixed length alone", () => {
    const fixed = [{ kind: "generated", min: 75, max: 75 }];
    expect(drawLengths(fixed, { run: 9, seed: 3 })).toEqual(fixed);
  });

  // 🔴 THE LIMIT IS GONE, which is the change: a range under Protein Hunter is
  // a job rather than a refusal.
  //
  // 🔴 AND IT DOES NOT UNBLOCK THE MOTIF PRESETS, which an earlier version of
  // this test claimed by handing `chainsOf` a `sequenceOf` that answered ANY
  // range. Measured against the real structures, those presets have a SECOND
  // and unrelated problem: their contig is written in the original PDB entry's
  // numbering (`A52-71`) while the shipped example CIF is a cropped copy whose
  // chain A runs 0-42. RFD4 never notices because a preset without `specMode`
  // runs from its structure's annotations rather than from its contig; Hunter
  // reads the contig literally and is right to refuse. A fixture that answers
  // every question proves nothing about the ones that have a real answer.
  it("no longer refuses a range under Protein Hunter", () => {
    expect(METHOD_LIMITS.hunter.range).toBe(undefined);
    const spans = [{ kind: "generated", min: 5, max: 20 }];
    expect(contigOf(spans, "hunter").problems).toEqual([]);
    const job = chainsOf(drawLengths(spans, { run: 0, seed: 0 }), {});
    expect(job.problems).toEqual([]);
    expect(job.length >= 5 && job.length <= 20).toBe(true);
  });
});
