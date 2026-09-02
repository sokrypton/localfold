/**
 * The entity list: what to fold, as AlphaFold Server models it.
 *
 * The conversions are the whole risk here. Everything below the page still
 * reads a colon-joined sequence and a list of CCD codes, so a copies count that
 * expands wrongly, or a ligand that lands before a polymer, is a batch whose
 * every shape is self-consistent and whose contents are somebody else's fold.
 */
import { describe, expect, it } from "./harness.js";
import {
  entitiesFromText, entitiesProblem, entityProblem, expandEntities, newEntity,
} from "../web/entities.js";

const protein = (value, copies = 1) => ({ type: "protein", value, copies });
const ligand = (value, copies = 1) => ({ type: "ligand", value, copies });

describe("entity validation", () => {
  it("accepts a protein and a CCD ligand", () => {
    expect(entityProblem(protein("ACDEFGHIKL"))).toBe(null);
    expect(entityProblem(ligand("HEM"))).toBe(null);
    expect(entityProblem(ligand("gol"))).toBe(null);
  });

  it("rejects a colon inside a protein row", () => {
    // 🔴 ONE ROW IS ONE CHAIN. Splitting silently would make copies ambiguous:
    // two copies of "A:B" is four chains in one of two different orders.
    expect(entityProblem(protein("ACDE:FGHI"))).toMatch(/one sequence per entity/i);
  });

  it("rejects a CCD code that is not one", () => {
    expect(entityProblem(ligand("HEMOGLOBIN"))).toMatch(/1-5 letters or digits/);
    expect(entityProblem(ligand("HE-M"))).toMatch(/1-5 letters or digits/);
  });

  it("names the empty field after its own type", () => {
    expect(entityProblem(protein(""))).toMatch(/protein sequence/);
    expect(entityProblem(ligand(""))).toMatch(/CCD code/);
  });

  it("rejects copies that are not a sensible count", () => {
    expect(entityProblem(protein("ACDE", 0))).toMatch(/at least 1/);
    expect(entityProblem(protein("ACDE", 2.5))).toMatch(/whole number/);
    expect(entityProblem(protein("ACDE", 999))).toMatch(/At most 20/);
  });

  it("allows a ligand on its own, which AF3 does too", () => {
    // 🔴 THIS USED TO BE REFUSED, on the assumption that every layer below
    // indexes on a polymer sequence. It does not: the chain identity helpers
    // reject a zero-length sequence, rightly, but they are read only inside the
    // polymer loop, which does not run when there are no residues.
    expect(entitiesProblem([ligand("HEM")])).toBe(null);
    expect(expandEntities([ligand("HEM")])).toEqual({
      chains: [], chainKinds: [], ligandCodes: ["HEM"], modifications: [],
      sequence: "",
    });
  });

  it("gives every copy of a chain its own modification, numbered by chain", () => {
    // 🔴 COPIES ARE EXPANDED, so two copies of a phosphorylated chain are two
    // chains each carrying it - and the featuriser indexes by the chain it will
    // actually see, which is the position in `chains` and not in `entities`.
    const modified = { type: "protein", value: "ACSEFG", copies: 2,
                       modifications: [{ code: "sep", position: 3 }] };
    const plain = { type: "protein", value: "MKV", copies: 1, modifications: [] };
    const out = expandEntities([modified, plain]);
    expect(out.chains).toEqual(["ACSEFG", "ACSEFG", "MKV"]);
    expect(out.modifications).toEqual([
      { chain: 0, position: 3, code: "SEP" },
      { chain: 1, position: 3, code: "SEP" },
    ]);
  });

  it("still needs something to fold", () => {
    expect(entitiesProblem([])).toMatch(/Add an entity/);
  });

  it("numbers the entity a problem is in, but only when there are several", () => {
    expect(entitiesProblem([protein("ACDE"), ligand("!!")])).toMatch(/^Entity 2: /);
    expect(entitiesProblem([ligand("!!")])).toMatch(/^A CCD code/);
  });

  it("makes a blank protein row by default", () => {
    // `modifications` is present and empty rather than absent, so every reader
    // can iterate it without asking whether it is there.
    expect(newEntity()).toEqual({ type: "protein", value: "", copies: 1, modifications: [] });
    expect(newEntity("ligand").type).toBe("ligand");
  });
});

describe("expanding entities for the fold pipeline", () => {
  it("turns copies into repeated chains and repeated ligands", () => {
    const { chains, ligandCodes, sequence } = expandEntities([
      protein("ACDEFGHIKL", 2), ligand("hem", 2),
    ]);
    expect(chains).toEqual(["ACDEFGHIKL", "ACDEFGHIKL"]);
    expect(ligandCodes).toEqual(["HEM", "HEM"]);
    expect(sequence).toBe("ACDEFGHIKL:ACDEFGHIKL");
  });

  it("puts every polymer before every ligand, whatever the entry order", () => {
    // 🔴 featuriseProtein appends ligand tokens AFTER all polymer tokens and
    // numbers asym_id straight on from the last chain. A ligand entered first
    // would otherwise claim a chain index the polymers still use.
    const { chains, ligandCodes } = expandEntities([
      ligand("ATP"), protein("ACDEFGHIKL"), ligand("HEM"),
    ]);
    expect(chains).toEqual(["ACDEFGHIKL"]);
    expect(ligandCodes).toEqual(["ATP", "HEM"]);
  });

  it("throws rather than expanding something invalid", () => {
    expect(() => expandEntities([protein("")])).toThrow(/protein sequence/);
  });
});

describe("entities from pasted text", () => {
  it("reads a bare sequence", () => {
    expect(entitiesFromText("  acdefghikl \n")).toEqual([protein("ACDEFGHIKL")]);
  });

  it("splits colon-separated chains into rows", () => {
    expect(entitiesFromText("ACDE:FGHI")).toEqual([protein("ACDE"), protein("FGHI")]);
  });

  it("collapses identical chains into copies", () => {
    // Which is what the featuriser concludes anyway - chainIdentity groups by
    // sequence - so the list now shows what will actually be folded.
    expect(entitiesFromText("ACDE:ACDE:FGHI")).toEqual([protein("ACDE", 2), protein("FGHI")]);
  });

  it("reads multi-record FASTA, dropping the description lines", () => {
    const fasta = ">first chain\nACDE\nFGHI\n>second\nKLMN\n";
    expect(entitiesFromText(fasta)).toEqual([protein("ACDEFGHI"), protein("KLMN")]);
  });

  it("never guesses a ligand", () => {
    // A bare CCD code is indistinguishable from a very short peptide, and
    // guessing wrong is worse than not guessing.
    expect(entitiesFromText("HEM")).toEqual([protein("HEM")]);
  });
});

describe("nucleic entities", () => {
  const chain = (type, value, copies = 1) => ({ type, value, copies, modifications: [] });

  it("carries a kind per chain, in the order the chains are in", () => {
    // 🔴 THE ONLY THING THAT SAYS WHAT A CHAIN'S LETTERS MEAN. `ACGT` is a
    // valid protein and a valid DNA chain, so the row's type has to travel with
    // it - and by CHAIN rather than by entity, because copies are expanded.
    const expanded = expandEntities([
      chain("protein", "ACDEFGHIKL"), chain("dna", "ACGT", 2), chain("rna", "ACGU"),
    ]);
    expect(expanded.chainKinds).toEqual(["protein", "dna", "dna", "rna"]);
    expect(expanded.chains).toEqual(["ACDEFGHIKL", "ACGT", "ACGT", "ACGU"]);
  });

  it("reads the same letters differently in a protein row and a DNA row", () => {
    expect(entityProblem(chain("protein", "ACGT"))).toBe(null);
    expect(entityProblem(chain("dna", "ACGT"))).toBe(null);
  });

  it("names the swapped base rather than listing the alphabet", () => {
    // A U in a DNA row is a pasted RNA sequence, not a typo for T, and the
    // message that helps says which row it belongs in.
    expect(entityProblem(chain("dna", "ACGU"))).toMatch(/RNA/);
    expect(entityProblem(chain("rna", "ACGT"))).toMatch(/DNA/);
  });

  it("refuses N, which has no nucleic slot in the alphabet", () => {
    // The restypes run A G C U and DA DG DC DT and stop; an unknown base would
    // have to borrow the amino-acid UNK.
    expect(entityProblem(chain("dna", "ACGN"))).toMatch(/N is not one of A, C, G, T/);
  });

  it("refuses a modified base rather than featurising it as an amino acid", () => {
    const modified = { ...chain("dna", "ACGT"),
      modifications: [{ code: "SEP", position: 2 }] };
    expect(entityProblem(modified)).toMatch(/not supported/);
  });
});
