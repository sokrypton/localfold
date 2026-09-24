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
  entitiesFromText, entitiesProblem, entityProblem, expandEntities, newEntity, parseContact, templateAsked, templateKind, templateProblem,
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
      chains: [], chainKinds: [],
      // ...and no chain wanting an alignment, for the same reason it takes
      // no template: there is no polymer here to align.
      chainMsa: [],
      ligandCodes: ["HEM"], modifications: [],
      // ...a ligand takes no template: AF3's own Template is one protein
      // chain, and a ligand has no residues to map onto.
      templates: [],
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

describe("an alignment turned off for one chain", () => {
  /**
   * 🔴 PER CHAIN, AND THE COPIES INHERIT IT. `chainMsa` sits beside
   * `chainKinds` in chain order because that is the only thing that says
   * which block belongs to which chain - the fold path filters it to the
   * protein chains and hands the mask to the search, which gives an
   * unwanted chain its query row alone.
   */
  it("marks the chains that asked for one, in chain order", () => {
    const out = expandEntities([
      { type: "protein", value: "GWSTELEKHRSVQ", copies: 2, modifications: [] },
      { type: "protein", value: "PIAQIHILEGRSD", copies: 1, msa: "none",
        modifications: [] },
      { type: "ligand", value: "HEM", copies: 1, modifications: [] },
    ]);
    expect(out.chainMsa).toEqual([true, true, false]);
  });

  it("says yes for an entity that says nothing, which is every older job", () => {
    const out = expandEntities([
      { type: "protein", value: "GWSTELEKHRSVQ", copies: 1, modifications: [] },
    ]);
    expect(out.chainMsa).toEqual([true]);
  });

  it("never asks for one for a chain that is not protein", () => {
    const out = expandEntities([
      { type: "dna", value: "ACGTACGT", copies: 1, modifications: [] },
      { type: "protein", value: "GWSTELEKHRSVQ", copies: 1, modifications: [] },
    ]);
    expect(out.chainMsa).toEqual([false, true]);
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

describe("entity templates", () => {
  const withTemplate = (source, extra = {}) => ({
    type: "protein", value: "ACDEFGHIK", copies: 1, modifications: [],
    template: { kind: "pdb", source, ...extra },
  });

  it("accepts a PDB entry and a chain suffix", () => {
    for (const source of ["1abc", "1abc_A", "1abc:B"]) {
      expect(templateProblem(withTemplate(source))).toBe(null);
    }
  });

  it("accepts an accession once the source says AlphaFold DB", () => {
    expect(templateProblem(withTemplate("P00533", { kind: "afdb" }))).toBe(null);
  });

  // 🔴 THE KIND IS WHAT MAKES THESE WRONG, and neither was wrong before it. A
  // four-character accession went to the PDB and a six-character entry to
  // AlphaFold DB, both silently, because the old field decided by counting.
  it("holds each source to its own spelling", () => {
    expect(templateProblem(withTemplate("P00533"))).toContain("not a PDB entry");
    expect(templateProblem(withTemplate("1abc", { kind: "afdb" })))
      .toContain("not a UniProt accession");
  });

  it("says so when the source is not one of those", () => {
    expect(templateProblem(withTemplate("not a code!"))).toContain("not a PDB entry");
  });

  // AF3's own Template is documented as one protein chain, and a ligand has no
  // residues for a map to name.
  it("refuses a template on anything but a protein", () => {
    expect(templateProblem({ ...withTemplate("1abc"), type: "ligand" }))
      .toContain("Only a protein chain");
  });

  it("ignores an empty source rather than complaining about it", () => {
    expect(templateProblem(withTemplate(""))).toBe(null);
    expect(templateProblem({ type: "protein", value: "AC", copies: 1 })).toBe(null);
  });

  // 🔴 AN UNFINISHED ROW IS NOT AN ERROR AND IS NOT A TEMPLATE EITHER. Picking
  // a source and not yet typing under it is the state the dropdown is in for as
  // long as it takes to type, so it cannot be a problem - but folding as though
  // it were a template would put an empty slot where a structure is expected.
  it("asks for a template only once the source has something under it", () => {
    expect(templateAsked({ kind: "pdb", source: "" })).toBe(false);
    expect(templateAsked({ kind: "pdb", source: "1abc" })).toBe(true);
    expect(templateAsked({ kind: "upload" })).toBe(false);
    expect(templateAsked({ kind: "upload", text: "ATOM..." })).toBe(true);
    // The search needs nothing typed: the hits are what it uses.
    expect(templateAsked({ kind: "search" })).toBe(true);
    expect(templateAsked({ kind: "none", source: "1abc" })).toBe(false);
    expect(templateAsked(undefined)).toBe(false);
  });

  it("treats an unknown kind as no template rather than trusting it", () => {
    expect(templateKind({ kind: "wikipedia", source: "1abc" })).toBe("none");
    expect(templateAsked({ kind: "wikipedia", source: "1abc" })).toBe(false);
  });

  // 🔴 PER CHAIN, NOT PER ENTITY, because copies are expanded - the same rule
  // the modifications follow, and for the same reason: the embedder indexes
  // slots by the chain number it will actually see.
  it("gives every copy of a chain its own template", () => {
    const expanded = expandEntities([
      { ...withTemplate("1abc_A"), copies: 2 },
      { type: "protein", value: "MKV", copies: 1, modifications: [] },
    ]);
    expect(expanded.templates.map((template) => template.chain)).toEqual([0, 1]);
    expect(expanded.templates[0].source).toBe("1abc_A");
  });

  it("expands an uploaded structure and a search the same way", () => {
    const expanded = expandEntities([
      { type: "protein", value: "ACDEF", copies: 1, modifications: [],
        template: { kind: "upload", text: "ATOM  ...", filename: "mine.pdb" } },
      { type: "protein", value: "MKVLA", copies: 1, modifications: [],
        template: { kind: "search" } },
    ]);
    expect(expanded.templates.map((template) => template.kind))
      .toEqual(["upload", "search"]);
  });
})

describe("mixed ligands in one job", () => {
  /**
   * 🔴 TWO DIFFERENT SMILES WERE BOTH CALLED `LIG`, AND THAT WAS TWO BUGS AT
   * ONCE. The output PDB wrote a benzene and a glycerol under one residue
   * name, which a reader cannot tell apart and which tooling that filters by
   * name silently mixes - and, worse, `featuriseProtein` keys a ligand's
   * ENTITY on its code, so the two came out sharing an `entity_id` with the
   * model told that six carbons and a glycerol are two copies of one thing.
   *
   * The names are distinct now and the featuriser keys on the molecule rather
   * than the name, which is belt and braces on purpose: either fix alone
   * leaves the other failure reachable from a different caller.
   */
  it("gives each distinct structure its own residue name", () => {
    const { ligandCodes } = expandEntities([
      { type: "smiles", value: "c1ccccc1", copies: 1, modifications: [] },
      { type: "smiles", value: "OCC(O)CO", copies: 1, modifications: [] },
    ]);
    expect(ligandCodes.map((entry) => entry.code)).toEqual(["LIG", "LG2"]);
  });

  it("gives identical structures the SAME name, because they are one entity", () => {
    const { ligandCodes } = expandEntities([
      { type: "smiles", value: "c1ccccc1", copies: 2, modifications: [] },
      { type: "smiles", value: "OCC(O)CO", copies: 1, modifications: [] },
      { type: "smiles", value: "c1ccccc1", copies: 1, modifications: [] },
    ]);
    expect(ligandCodes.map((entry) => entry.code)).toEqual(["LIG", "LIG", "LG2", "LIG"]);
  });

  it("keeps every name inside the PDB's three characters", () => {
    // 🔴 `src/af3/fold.js` writes the residue name with `.padEnd(3)` into a
    // fixed-width field, so a four-character name runs into the chain id - and
    // `LIG2` truncated back to `LIG` would put the collision straight back.
    const rows = [];
    for (let index = 0; index < 40; index += 1) {
      rows.push({ type: "smiles", value: `${"C".repeat(index + 1)}O`,
                  copies: 1, modifications: [] });
    }
    const codes = expandEntities(rows).ligandCodes.map((entry) => entry.code);
    expect(codes.every((code) => code.length === 3)).toBe(true);
    expect(new Set(codes).size).toBe(40);
  });

  it("mixes a SMILES with a CCD code without confusing them", () => {
    const { ligandCodes } = expandEntities([
      { type: "protein", value: "MKTSYIAKQRQ", copies: 1, modifications: [] },
      { type: "smiles", value: "c1ccccc1", copies: 1, modifications: [] },
      { type: "ligand", value: "atp", copies: 1, modifications: [] },
    ]);
    // The CCD code is upper-cased and stays a string; the SMILES is neither.
    expect(ligandCodes).toEqual([{ smiles: "c1ccccc1", code: "LIG" }, "ATP"]);
  });
});

/**
 * 🔴 A `contact` ROW IS A BOND, AND THE POINT IS THAT IT IS A ROW. AlphaFold 3's
 * `bondedAtomPairs` was read into a page variable first - state nobody could
 * see, invalidated by any edit to the chains it named, which is exactly what
 * the job NAME was built and then removed for. A contact sits in the entity
 * list with those chains: visible, editable, and gone when the reader deletes
 * it.
 *
 * Measured end to end on AlphaFold 3's own KRAS/sotorasib example, which
 * declares the covalent bond to cysteine 12: SG-C25 is **1.62 A** folded from
 * the row and **6.25 A** with the bond removed. See docs/WEB.md.
 */
describe("a contact row", () => {
  const contact = (value) => ({ type: "contact", value, copies: 1, modifications: [] });

  it("reads chains, residues and optional atoms", () => {
    expect(parseContact("A12:SG - B1:C25")).toEqual({
      from: { chain: "A", residue: 12, atom: "SG" },
      to: { chain: "B", residue: 1, atom: "C25" },
    });
    expect(parseContact("A12 - B30")).toEqual({
      from: { chain: "A", residue: 12, atom: null },
      to: { chain: "B", residue: 30, atom: null },
    });
  });

  it("refuses what is not a contact, by message", () => {
    expect(entityProblem(contact("nonsense"))).toMatch(/two residues/);
    expect(entityProblem(contact("A12 - A12"))).toMatch(/two different residues/);
    expect(entityProblem(contact("A12:SG - B1:C25"))).toBe(null);
  });

  /**
   * 🔴 IT IS NOT A CHAIN, AND `expandEntities` WOULD HAVE MADE IT A LIGAND.
   * That loop is `if polymer ... else if smiles ... else LIGAND`, so a row type
   * it has not heard of becomes a ligand - silently, with a chain in the fold
   * nobody asked for. The same shape as `setChains` deleting a SMILES row by
   * keeping only `type === "ligand"`.
   */
  it("adds no chain and no ligand", () => {
    const out = expandEntities([
      { type: "protein", value: "GWCTELEKH", copies: 1, modifications: [] },
      contact("A3:SG - A7:SG"),
    ]);
    expect(out.chains).toHaveLength(1);
    expect(out.ligandCodes).toHaveLength(0);
  });

  /**
   * 🔴 THE LETTERS RESOLVE AGAINST THE ORDER THE FOLD SEES: polymer chains
   * first, then each ligand, which is how `featuriseProtein` assigns `asymId`.
   * One number reaches the featuriser and no second convention is invented.
   */
  it("resolves its letters to the fold's own chain numbering", () => {
    const out = expandEntities([
      { type: "protein", value: "GWCTELEKH", copies: 1, modifications: [] },
      { type: "ligand", value: "GOL", copies: 1, modifications: [] },
      contact("A3:SG - B1:C1"),
    ]);
    expect(out.bonds).toEqual([{
      from: { asym: 0, residue: 3, atom: "SG" },
      to: { asym: 1, residue: 1, atom: "C1" },
    }]);
  });
});
