/**
 * DNA and RNA chains, at the seams a unit test can hold and the oracle cannot.
 *
 * The authority is tools/oracle/check_af3_featurise.js with --dna/--rna, which
 * reproduces AF3's batch array by array for a protein+DNA, a protein+RNA and a
 * three-chain complex. What it cannot check is the things that are true of the
 * PORT rather than of AF3: that a chain's kind is what decides how its letters
 * are read, and that what comes out the other end is written as a nucleic acid
 * rather than as the amino acids those letters also spell.
 *
 * 🔴 THE FAILURE MODE HERE IS A PLAUSIBLE ANSWER. `ACGT` is a valid protein and
 * a valid DNA chain, so every one of these can be wrong without erroring: a
 * DNA chain read as protein folds to a four-residue peptide, and one WRITTEN as
 * protein draws as a ribbon. Both look like a fold.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { featuriseProtein } from "../src/af3/featurise.js";
import { toPdb } from "../src/af3/fold.js";
import { nucleicAatypeFor } from "../src/af3/reference-conformers-nucleic.js";

const blank = (batch) => new Float32Array(batch.tokens * batch.dense * 3);

describe("nucleic chains", () => {
  it("reads the same letters as bases or as amino acids, by the chain's kind", () => {
    // Measured against AF3: DNA ACGT is aatype 26 28 27 29, and the same
    // letters in a protein chain are 0 4 7 16.
    const dna = featuriseProtein("ACGT", { chainKinds: ["dna"] });
    assert.deepEqual(Array.from(dna.aatype), [26, 28, 27, 29]);
    const protein = featuriseProtein("ACGT");
    assert.deepEqual(Array.from(protein.aatype), [0, 4, 7, 16]);
  });

  it("is one token per residue, unlike a ligand or a modified residue", () => {
    // This is why nucleic acids needed no tokeniser change and modified
    // residues did.
    const both = featuriseProtein("ACDEFGHIKL:ACGT:ACGU",
      { chainKinds: ["protein", "dna", "rna"] });
    assert.equal(both.tokens, 18);
  });

  it("defaults every chain to protein, which is what callers before this meant", () => {
    const implied = featuriseProtein("ACGT");
    const stated = featuriseProtein("ACGT", { chainKinds: ["protein"] });
    assert.deepEqual(Array.from(implied.aatype), Array.from(stated.aatype));
  });

  it("takes the extra terminal atom at the 5' end, mirroring the protein's OXT", () => {
    // A protein gains OXT on its LAST residue; a nucleotide gains OP3 on its
    // FIRST. Reusing the protein rule would put it on the wrong base.
    const dna = featuriseProtein("ACGT", { chainKinds: ["dna"] });
    const atomsOf = (token) => {
      let count = 0;
      for (let atom = 0; atom < dna.dense; atom += 1) {
        if (dna.refMask[token * dna.dense + atom]) count += 1;
      }
      return count;
    };
    // Measured against AF3: 22 19 22 20, the first base one atom above internal.
    assert.deepEqual([0, 1, 2, 3].map(atomsOf), [22, 19, 22, 20]);
  });

  it("writes DNA as ' DA' and RNA as '  A', which is what tells them apart", () => {
    // 🔴 THREE_LETTER WOULD MAKE THIS A POLY-ALANINE. `A` is ALA and `G` is GLY
    // in the amino-acid table, so a DNA chain written through it is a peptide
    // as far as any viewer or scoring tool is concerned.
    const batch = featuriseProtein("ACGT:ACGU", { chainKinds: ["dna", "rna"] });
    const pdb = toPdb(batch, blank(batch), null);
    const names = new Set();
    for (const line of pdb.split("\n")) {
      if (line.startsWith("ATOM") || line.startsWith("HETATM")) {
        names.add(`${line.slice(21, 22)}${JSON.stringify(line.slice(17, 20))}`);
      }
    }
    for (const want of ['A" DA"', 'A" DT"', 'B"  A"', 'B"  U"']) {
      assert.ok(names.has(want), `${want} missing from ${[...names].join(" ")}`);
    }
  });

  it("agrees with the conformer table about the alphabet", () => {
    // The one place the aatypes are written down twice; they must not drift.
    const dna = featuriseProtein("ACGT", { chainKinds: ["dna"] });
    for (const [index, code] of [..."ACGT"].entries()) {
      assert.equal(dna.aatype[index], nucleicAatypeFor("dna", code));
    }
  });

  it("gaps a nucleic token in a protein alignment row, and keeps it in the query", () => {
    // Measured against AF3: rows 0 and 1 read the DNA aatype at the DNA
    // columns and every row below reads MSA_GAP.
    const batch = featuriseProtein("ACDEFGHIKL:ACGT", {
      chainKinds: ["protein", "dna"],
      msa: [Int32Array.from([0, 4, 3, 6, 13, 7, 8, 9, 11, 10]),
            Int32Array.from([0, 4, 3, 21, 13, 7, 8, 9, 11, 19])],
    });
    const row = (index) => Array.from(batch.msa.slice(index * batch.tokens,
                                                      (index + 1) * batch.tokens));
    assert.deepEqual(row(0).slice(10), [26, 28, 27, 29]);   // the query
    assert.deepEqual(row(1).slice(10), [26, 28, 27, 29]);   // its own alignment
    assert.deepEqual(row(2).slice(10), [21, 21, 21, 21]);   // a protein homolog
  });

  it("gives a nucleic token a profile of its own base, not of the gap", () => {
    const batch = featuriseProtein("ACDEFGHIKL:ACGT", {
      chainKinds: ["protein", "dna"],
      msa: [Int32Array.from([0, 4, 3, 21, 13, 7, 8, 9, 11, 19])],
    });
    const restypes = batch.profile.length / batch.tokens;
    // Token 10 is the DNA chain's first base: adenine, restype 26.
    assert.equal(batch.profile[10 * restypes + 26], 1);
  });
});
