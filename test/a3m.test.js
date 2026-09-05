import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { describe, expect, it } from "./harness.js";
import { parseA3m } from "../src/input/a3m.js";

describe("A3M parser", () => {
  it("parses the uploaded homolog-rich alignment", async() => {
    const text = await readFile(resolve("tools/fixtures/test.a3m"), "utf8");
    const alignment = parseA3m(text);
    expect(alignment.depth).toBe(8076);
    expect(alignment.length).toBe(59);
    expect(alignment.query).toBe("PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK");
    expect(alignment.deletionMatrix[0]).toEqual(new Array(59).fill(0));
  });

  it("accepts a query-only alignment", () => {
    const alignment = parseA3m(">query\nACDEFGHIK\n");
    expect(alignment.depth).toBe(1);
    expect(alignment.query).toBe("ACDEFGHIK");
  });

  it("removes lowercase insertions and records their counts", () => {
    const alignment = parseA3m(">query\nACDE\n>hit\nAcgC-E\n");
    expect(alignment.sequences).toEqual(["ACDE", "AC-E"]);
    expect(alignment.deletionMatrix).toEqual([
      [0, 0, 0, 0],
      [0, 2, 0, 0],
    ]);
  });

  it("supports multiline FASTA records and comments", () => {
    const alignment = parseA3m("# metadata\n>query\nAC\nDE\n>hit\nAC\n-E\n");
    expect(alignment.sequences).toEqual(["ACDE", "AC-E"]);
  });

  it("rejects rows with inconsistent aligned lengths", () => {
    expect(() => parseA3m(">query\nACDE\n>bad\nACD\n")).toThrow(/aligned length 3; expected 4/);
  });

  // 🔴 A ROW LONGER THAN ONE CHUNK. The residues are assembled from a code
  // buffer in blocks of 4096, because spreading a whole row into
  // String.fromCharCode's arguments blows the stack on a long enough one - and
  // a3m rows are not bounded by anything this module controls. A row that
  // straddles the boundary is the case the chunking exists for.
  it("assembles a row longer than the chunk it is built in", () => {
    const query = "AC".repeat(3000);                              // 6000 residues
    const alignment = parseA3m(`>query\n${query}\n>hit\n${query}\n`);
    expect(alignment.length).toEqual(6000);
    expect(alignment.sequences[0]).toEqual(query);
    expect(alignment.sequences[1]).toEqual(query);
  });

  it("rejects a residue outside the aligned alphabet", () => {
    expect(() => parseA3m(">query\nACDE\n>bad\nAC*E\n"))
      .toThrow(/invalid residue "\*"/);
    // ...including one past the code table's end, which is a separate branch.
    expect(() => parseA3m(">query\nACDE\n>bad\nAC\u00c5E\n"))
      .toThrow(/invalid residue/);
  });
});
