import { describe, expect, it } from "./harness.js";
import {
  alanineBias, alanineFraction, designBias, omittedLetters, sampleSequence,
} from "../src/design/sample-sequence.js";
import { ALPHABET } from "../src/design/mpnn/constants.js";
import { uniformFrom } from "../src/af3/fold.js";

/** A deterministic stand-in for Math.random that cycles a fixed list. */
const cycling = (values) => {
  let at = 0;
  return () => values[(at += 1) % values.length];
};

describe("sampleSequence", () => {
  it("makes every position unknown at 100 percent", () => {
    const sequence = sampleSequence(24, { percentX: 100, random: cycling([0.5]) });
    expect(sequence).toBe("X".repeat(24));
  });

  it("makes none unknown at 0 percent, and draws only real letters", () => {
    const sequence = sampleSequence(40, { percentX: 0, random: uniformFrom(7) });
    expect(sequence).toHaveLength(40);
    expect(/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(sequence)).toBe(true);
  });

  // The reference is `round(length * frac_X)`, not floor: 90% of 15 is 13.5,
  // which rounds up. A floor here would put one more designable position in
  // every short binder than the paper's method does.
  it("rounds the unknown count rather than truncating it", () => {
    const sequence = sampleSequence(15, { percentX: 90, random: cycling([0.5]) });
    expect([...sequence].filter((letter) => letter === "X")).toHaveLength(14);
  });

  // See the note on POOL_WITHOUT_PROLINE: the flag ADDS proline when false,
  // because the reference's literal never contained it.
  it("draws proline only when excludeP is false", () => {
    const withP = sampleSequence(400, { percentX: 0, excludeP: false, random: uniformFrom(3) });
    const withoutP = sampleSequence(400, { percentX: 0, excludeP: true, random: uniformFrom(3) });
    expect(withP).toContain("P");
    expect(withoutP.includes("P")).toBe(false);
  });

  it("shuffles, so the unknown positions are not a prefix", () => {
    const sequence = sampleSequence(60, { percentX: 50, random: uniformFrom(11) });
    expect(sequence.slice(0, 30)).toContain("X");
    // ...and the count survives the shuffle.
    expect([...sequence].filter((letter) => letter === "X")).toHaveLength(30);
  });

  it("is reproducible from a seed", () => {
    expect(sampleSequence(50, { random: uniformFrom(19) }))
      .toBe(sampleSequence(50, { random: uniformFrom(19) }));
  });
});

describe("alanineBias", () => {
  it("ramps linearly from start to end across the run", () => {
    expect(alanineBias(0, 5)).toBeCloseTo(-0.5, 6);
    expect(alanineBias(4, 5)).toBeCloseTo(-0.1, 6);
    expect(alanineBias(2, 5)).toBeCloseTo(-0.3, 6);
  });

  it("takes the start value when there is nothing to ramp across", () => {
    expect(alanineBias(0, 1)).toBeCloseTo(-0.5, 6);
  });

  it("honours overridden endpoints", () => {
    expect(alanineBias(1, 3, { start: -1, end: 1 })).toBeCloseTo(0, 6);
  });
});

describe("omittedLetters", () => {
  it("adds proline on the first design cycle only", () => {
    expect([...omittedLetters(0)].sort().join("")).toBe("CP");
    expect(omittedLetters(1)).toBe("C");
    expect(omittedLetters(4)).toBe("C");
  });

  it("keeps every standing omission and deduplicates", () => {
    expect([...omittedLetters(0, "C,W")].sort().join("")).toBe("CPW");
    expect([...omittedLetters(0, "CP")].sort().join("")).toBe("CP");
  });

  it("has no standing omission to add to when given none", () => {
    expect(omittedLetters(1, "")).toBe("");
  });
});

describe("designBias", () => {
  const row = (bias, position = 0) =>
    bias.slice(position * ALPHABET.length, (position + 1) * ALPHABET.length);

  it("always omits X, which is an input letter and never a design choice", () => {
    expect(row(designBias(3))[ALPHABET.indexOf("X")]).toBe(-1e8);
  });

  it("omits the letters it is given and leaves the rest alone", () => {
    const first = row(designBias(2, { omit: "CP" }));
    expect(first[ALPHABET.indexOf("C")]).toBe(-1e8);
    expect(first[ALPHABET.indexOf("P")]).toBe(-1e8);
    expect(first[ALPHABET.indexOf("A")]).toBe(0);
  });

  it("repeats one row across every position", () => {
    const bias = designBias(4, { omit: "C" });
    expect(bias).toHaveLength(4 * ALPHABET.length);
    for (let position = 1; position < 4; position += 1) {
      expect(row(bias, position)).toEqual(row(bias, 0));
    }
  });

  // The two compose rather than overwrite: "omit alanine" has to survive a
  // ramp that would otherwise assign over it.
  it("adds the alanine bias without lifting an alanine omission", () => {
    expect(row(designBias(1, { alanineBias: -0.4 }))[ALPHABET.indexOf("A")])
      .toBeCloseTo(-0.4, 6);
    expect(row(designBias(1, { omit: "A", alanineBias: -0.4 }))[ALPHABET.indexOf("A")])
      .toBeLessThan(-1e7);
  });
});

describe("alanineFraction", () => {
  it("counts alanine", () => {
    expect(alanineFraction("AAAA")).toBe(1);
    expect(alanineFraction("AGAG")).toBe(0.5);
    expect(alanineFraction("GGGG")).toBe(0);
  });

  it("is zero for the empty sequence rather than NaN", () => {
    expect(alanineFraction("")).toBe(0);
  });
});
