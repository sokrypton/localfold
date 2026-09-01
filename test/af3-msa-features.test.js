/**
 * A3M text into AF3's MSA codes.
 *
 * The oracle (tools/oracle/check_af3_featurise.js, with --a3m) checks this
 * against AF3's own batch and is the authority. What it cannot check is the
 * parts of the alphabet a real alignment does not happen to contain: MMseqs2
 * does not emit B, Z, J, O or U, so the ambiguity aliases would go untested by
 * any number of exact matches on real data. Those are here, along with the
 * boundaries - the gap's index and the depth cap - stated as constants rather
 * than derived, so a change to either has to be a change to this file.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { af3MsaFromA3m, AF3_MSA_CODES, AF3_MSA_GAP } from "../src/af3/msa-features.js";

const QUERY = "ACDEFGHIKL";
const a3m = (...rows) => rows.map((row, index) => `>seq${index}\n${row}`).join("\n") + "\n";

describe("af3MsaFromA3m", () => {
  it("puts the gap at 21, between the amino acids and the nucleotides", () => {
    // 🔴 THE ONE CONSTANT MOST LIKELY TO BE WRONG. AF3's MSA one-hot is 32 wide
    // and the gap is NOT at the end of it: the alphabet is 21 protein codes,
    // then the gap, then the nucleotides. A gap at 31 would fold and be wrong.
    assert.equal(AF3_MSA_GAP, 21);
    assert.equal(AF3_MSA_CODES["-"], 21);
    const { msa } = af3MsaFromA3m(a3m(QUERY, "AC--FGHIKL"));
    assert.deepEqual(Array.from(msa[1]).slice(0, 5), [0, 4, 21, 21, 13]);
  });

  it("keeps the A3M's own query row, which the model sees twice", () => {
    // See the note in msa-features.js: AF3's msa is the paired block followed
    // by the unpaired one, and with no pairing the paired block is the query.
    const rows = af3MsaFromA3m(a3m(QUERY, "AC--FGHIKL"));
    assert.equal(rows.msa.length, 2, "both A3M rows, query included");
    assert.equal(rows.depth, 3, "and a third row prepended downstream");
    assert.deepEqual(Array.from(rows.msa[0]), Array.from(rows.msa[0]).map((_, index) =>
      AF3_MSA_CODES[QUERY[index]]));
  });

  it("counts deletions raw, not squashed", () => {
    // AF2 stores atan(n/3)*2/pi; AF3's embedder does that itself, from an
    // integer. Three insertions before the third column must arrive as 3.
    const { deletionMatrix } = af3MsaFromA3m(a3m(QUERY, "ACqrsDEFGHIKL"));
    assert.deepEqual(Array.from(deletionMatrix[1]).slice(0, 4), [0, 0, 3, 0]);
  });

  it("resolves the ambiguity codes as AF3 does", () => {
    // B and Z to the acid rather than the amide, U to cysteine, J and O to
    // unknown.
    //
    // 🔴 ASSERTED ON THE TABLE, NOT THROUGH A3M TEXT, BECAUSE THE PARSER IS
    // NARROWER THAN THE TABLE. src/input/a3m.js accepts only ACDEFGHIKLMNPQRST
    // VWYX and the gap, so a row carrying any of these five is rejected before
    // this module sees it - for AlphaFold 2 as well, which is why widening it
    // is not something the AF3 path should do on its own. The entries are kept
    // because they are what AF3 means by those letters, and the day the parser
    // admits them the answer should already be right rather than newly decided.
    assert.equal(AF3_MSA_CODES.B, AF3_MSA_CODES.D);
    assert.equal(AF3_MSA_CODES.Z, AF3_MSA_CODES.E);
    assert.equal(AF3_MSA_CODES.U, AF3_MSA_CODES.C);
    assert.equal(AF3_MSA_CODES.J, AF3_MSA_CODES.X);
    assert.equal(AF3_MSA_CODES.O, AF3_MSA_CODES.X);
  });

  it("caps the depth at what the model will read, counting the query", () => {
    // maxSequences is the model's num_msa, so the rows returned are one fewer:
    // featuriseProtein contributes the first. Off by one here is a silently
    // deeper or shallower MSA than the caller asked for.
    const text = a3m(QUERY, ...Array.from({ length: 20 }, () => "AC--FGHIKL"));
    assert.equal(af3MsaFromA3m(text, { maxSequences: 5 }).msa.length, 4);
    assert.equal(af3MsaFromA3m(text, { maxSequences: 5 }).depth, 5);
    // An alignment shallower than the cap is not padded up to it.
    assert.equal(af3MsaFromA3m(text, { maxSequences: 500 }).depth, 22);
  });

  it("puts the paired block first and the unpaired block after it", () => {
    // AF3's msa is paired ++ unpaired. Neither block contributes its own query
    // row when both are present: featuriseProtein writes row 0 from the
    // sequence, and AF3's deduplication removes the unpaired copy because a
    // paired alignment always contains the query.
    const paired = a3m(QUERY, "AAAAAAAAAA");
    const unpaired = a3m(QUERY, "CCCCCCCCCC");
    const rows = af3MsaFromA3m({ paired, unpaired });
    // 🔴 BOTH BLOCKS KEEP THEIR OWN QUERY ROW. AF3's pairing prepends a query
    // row and then emits rows that begin at the A3M's query again, exactly as
    // the unpaired block does - so each block contributes all of its A3M, and
    // only the row featuriseProtein writes is not repeated here. QUERY starts
    // with A, which is why three of these four are A.
    assert.deepEqual(rows.msa.map((row) => row[0]), [
      AF3_MSA_CODES.A,  // the paired block's query row
      AF3_MSA_CODES.A,  // and its one homolog
      AF3_MSA_CODES.C,  // the unpaired block's homolog - its query is the dup
    ]);
    // The profile is over the unpaired block alone: row 0 is the query
    // featuriseProtein writes, rows 1-2 are the paired block, so it starts at 3.
    assert.equal(rows.unpairedFrom, 3);
    assert.equal(rows.depth, 4);
  });

  it("points the profile at the query when there is no alignment at all", () => {
    const rows = af3MsaFromA3m({ paired: null, unpaired: null });
    assert.deepEqual(rows.msa, []);
    assert.equal(rows.depth, 1);
    assert.equal(rows.unpairedFrom, 0, "row zero, the query itself");
  });

  it("gives the paired block at most half the budget, and the rest to unpaired", () => {
    // 🔴 THE REGRESSION THIS FILE EXISTED WITHOUT. AF3 sets
    // max_paired_sequences = msa_size // 2 and then fills the remainder from the
    // unpaired block. Letting the paired block take the whole cap is the natural
    // way to write the loop, and it only shows on a HOMO-OLIGOMER: a monomer has
    // no paired block, so every single-chain check passes while a dimer folds
    // against a paired block that has eaten the entire budget, no unpaired rows
    // at all, and a profile with nothing left to average.
    const deep = (residue) => a3m(QUERY, ...Array.from({ length: 40 }, () => residue.repeat(10)));
    const rows = af3MsaFromA3m({ paired: deep("A"), unpaired: deep("C") },
                               { maxSequences: 10 });
    assert.equal(rows.depth, 10, "the cap is still the cap");
    // The paired crop is 5 - half of 10 - of which row zero is the query
    // featuriseProtein writes, so four rows come from here. The unpaired crop
    // is the remaining 5, and it spends all five on HOMOLOGS: its own query row
    // is the duplicate AF3's deduplication removes, and the crop counts rows,
    // so skipping it moves the end rather than costing a row.
    assert.equal(rows.unpairedFrom, 5);
    assert.deepEqual(rows.msa.map((row) => row[0]), [
      ...Array(4).fill(AF3_MSA_CODES.A),
      ...Array(5).fill(AF3_MSA_CODES.C),
    ]);
    // ...so the profile sees five rows, not zero, which is the actual failure.
    assert.equal(rows.depth - rows.unpairedFrom, 5);
  });

  it("gives the unpaired block everything when there is no paired one", () => {
    const deep = a3m(QUERY, ...Array.from({ length: 40 }, () => "CCCCCCCCCC"));
    const rows = af3MsaFromA3m({ paired: null, unpaired: deep }, { maxSequences: 10 });
    assert.equal(rows.depth, 10);
    assert.equal(rows.unpairedFrom, 1, "straight after the query row");
  });

  it("takes the first rows rather than sampling them", () => {
    // AF3's truncate_msa_batch is jnp.arange(num_msa). MMseqs2 returns hits in
    // decreasing similarity, so the first N are the N most similar; a shuffle
    // would be a different model input, not a fairer sample.
    const text = a3m(QUERY, "AAAAAAAAAA", "CCCCCCCCCC", "DDDDDDDDDD");
    const { msa } = af3MsaFromA3m(text, { maxSequences: 3 });
    assert.deepEqual(Array.from(msa[1]).slice(0, 2), [AF3_MSA_CODES.A, AF3_MSA_CODES.A]);
  });
});
