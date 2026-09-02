import { describe, expect, it } from "./harness.js";
import {
  cleanSequence, cleanSequenceMap, complexSequenceProblem, extractFastaHeader,
  sequenceChains, sequenceProblem,
} from "../web/sequence.js";

describe("cleaning a pasted sequence", () => {
  it("leaves a plain sequence alone", () => {
    expect(cleanSequence("ACDEFGHIKL")).toBe("ACDEFGHIKL");
  });

  it("removes line breaks, tabs and spaces", () => {
    expect(cleanSequence("ACDE\nFGHI\r\nKL")).toBe("ACDEFGHIKL");
    expect(cleanSequence("ACDE\tFGHI KL")).toBe("ACDEFGHIKL");
    expect(cleanSequence("  ACDE  FGHI  ")).toBe("ACDEFGHIKL".slice(0, 8));
  });

  it("removes the position numbers a database prints down the margin", () => {
    // ...the shape UniProt and PDB use: a number every ten residues
    expect(cleanSequence("1 ACDEFGHIKL 11 MNPQRSTVWY"))
      .toBe("ACDEFGHIKLMNPQRSTVWY");
    expect(cleanSequence("   1  ACDEF GHIKL\n  11  MNPQR STVWY"))
      .toBe("ACDEFGHIKLMNPQRSTVWY");
  });

  it("drops a FASTA header without dropping the sequence under it", () => {
    expect(cleanSequence(">sp|P12345|SOME_PROT Description here\nACDEFGHIKL"))
      .toBe("ACDEFGHIKL");
    expect(cleanSequence("; a comment\nACDEF\n; another\nGHIKL")).toBe("ACDEFGHIKL");
  });

  it("removes alignment gaps and a trailing stop", () => {
    expect(cleanSequence("ACDE--FGHI...KL*")).toBe("ACDEFGHIKL");
  });

  it("uppercases", () => {
    expect(cleanSequence("acdefGHIkl")).toBe("ACDEFGHIKL");
  });

  it("handles the whole mess at once", () => {
    const pasted = ">sp|P0DTC2|SPIKE Fragment\n"
      + "   1  acdef ghikl\n"
      + "  11  MNPQR STVWY\n";
    expect(cleanSequence(pasted)).toBe("ACDEFGHIKLMNPQRSTVWY");
  });

  // 🔴 A MULTI-FASTA IS NOT ONE PROTEIN. Dropping every header and joining what
  // was left glued two sequences end to end and folded the chimera without a
  // word. This page folds one sequence, so several means the first.
  it("takes only the first record from a multi-FASTA", () => {
    expect(cleanSequence(">first\nACDEF\n>second\nWWWWW")).toBe("ACDEF");
    expect(cleanSequence(">a\nACDE\nFGHI\n>b\nWWWW")).toBe("ACDEFGHI");
  });

  it("treats `;` as a comment, not as a new record", () => {
    // ...they are different things in FASTA, and conflating them truncated a
    // sequence at its second comment line
    expect(cleanSequence("; note\nACDEF\n; another\nGHIKL")).toBe("ACDEFGHIKL");
    expect(cleanSequence(">a\n; note\nACDEF\nGHIKL")).toBe("ACDEFGHIKL");
  });

  it("keeps letters it does not recognise, so they can be reported", () => {
    // 🔴 NOT DROPPED. Silently removing an unexpected letter would fold a
    // different protein than the one that was pasted, without saying so.
    expect(cleanSequence("ACDEBFGHI")).toBe("ACDEBFGHI");
  });

  it("gives back nothing for input that was all formatting", () => {
    expect(cleanSequence("")).toBe("");
    expect(cleanSequence("   \n\t 123 \n")).toBe("");
    expect(cleanSequence(">just a header")).toBe("");
  });

  it("extracts FASTA header when present", () => {
    expect(extractFastaHeader("ACDEFGHIKL")).toBe(null);
    expect(extractFastaHeader(">my_protein\nACDEFGHIKL")).toBe("my_protein");
    expect(extractFastaHeader("  >  sp|P12345|MY_PROT some desc  \nACDEF")).toBe("sp|P12345|MY_PROT some desc");
    expect(extractFastaHeader("; comment\n>prot\nACDE")).toBe("prot");
    expect(extractFastaHeader(">\nACDE")).toBe(null);
  });
});

describe("reporting why a sequence cannot be folded", () => {
  it("passes the twenty and X", () => {
    expect(sequenceProblem("ARNDCQEGHILKMFPSTWYVX")).toBe(null);
  });

  it("asks for a sequence when there is none", () => {
    expect(sequenceProblem("")).toBe("Enter a protein sequence");
  });

  it("names the letter that is wrong, not just that something is", () => {
    expect(sequenceProblem("ACDEB")).toBe("B is not one of the twenty amino acids");
    expect(sequenceProblem("ACDEBZ")).toBe("B, Z are not among the twenty amino acids");
  });

  it("names each offending letter once, however often it appears", () => {
    expect(sequenceProblem("BBBACDEB")).toBe("B is not one of the twenty amino acids");
  });
});

describe("colon-separated complex sequences", () => {
  it("keeps boundaries while cleaning and returns physical chains", () => {
    const cleaned = cleanSequence("ACD : efG\n:ACD");
    expect(cleaned).toBe("ACD:EFG:ACD");
    expect(sequenceChains(cleaned)).toEqual(["ACD", "EFG", "ACD"]);
  });

  it("reports empty and invalid chains by position", () => {
    expect(complexSequenceProblem("ACD::EFG")).toMatch(/Every chain/);
    expect(complexSequenceProblem("ACD:EBG")).toBe("Chain 2: B is not one of the twenty amino acids");
  });
});

describe("mapping residues back to the box", () => {
  // 🔴 A RESIDUE'S POSITION IS NOT ITS OFFSET IN THE BOX. Headers, comment
  // lines and whitespace all go, so residue 12 of a pasted record can be
  // character 80 on screen - and anything pointing AT a residue needs to know
  // which. The map has to agree with cleanSequence exactly, or a highlight
  // lands on the wrong letter.
  const cases = ["ACDEF", "AC  DE", "AC\t\t DE", "A1234C", "AC DE\nF",
                 ">header\nACDE\nFGH", ";note\nACD\n>next\nZZZ", "", "   ",
                 "ac-de*f"];

  it("agrees with cleanSequence on every shape of input", () => {
    for (const text of cases) {
      expect(cleanSequenceMap(text).cleaned).toBe(cleanSequence(text));
    }
  });

  it("points every residue at the character it came from", () => {
    for (const text of cases) {
      const { cleaned, offsets } = cleanSequenceMap(text);
      expect(offsets.length).toBe(cleaned.length);
      offsets.forEach((offset, index) => {
        expect(text[offset].toUpperCase()).toBe(cleaned[index]);
      });
    }
  });

  it("does not let a run of spaces alternate", () => {
    // FORMATTING carries the `g` flag for its replace, and a global regex
    // remembers where it got to: testing one character at a time down a run of
    // spaces returned true, false, true, false, and "AC  DE" cleaned to
    // "AC DE" here while cleanSequence gave "ACDE".
    expect(cleanSequenceMap("AC    DE").cleaned).toBe("ACDE");
    expect(cleanSequenceMap("A1234C").cleaned).toBe("AC");
  });
});
