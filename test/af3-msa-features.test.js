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

import { af3MsaFromA3m, AF3_MSA_CODES, AF3_MSA_GAP, AF3_MSA_SIZE } from "../shared/af3/featurise/msa-features.js";

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

  it("clips a row's deletion count at 127, AF3's int8 - and the profile's rows keep the true count", () => {
    // features.py casts `deletion_matrix` with safe_cast_int8 after computing deletion_mean from the true counts
    const { deletionMatrix, profileDeletionMatrix } = af3MsaFromA3m(a3m(QUERY, "AC" + "q".repeat(300) + "DEFGHIKL"));
    assert.equal(deletionMatrix[1][2], 127);
    assert.equal(profileDeletionMatrix[1][2], 300);
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
    // NARROWER THAN THE TABLE. shared/input/a3m.js accepts only ACDEFGHIKLMNPQRST
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

  it("returns every row, not a draw: the trunk draws num_msa of them afresh each pass", () => {
    // AF3's featuriser keeps up to 16384 rows and its trunk shuffles and truncates to num_msa inside every pass
    // (shared/af3/msa-draw.js); one draw here, for the whole fold, was 8S83's gap to stock AF3. The rows returned
    // are one fewer than the depth: featuriseProtein contributes the first.
    const text = a3m(QUERY, ...Array.from({ length: 20 }, () => "AC--FGHIKL"));
    assert.equal(af3MsaFromA3m(text).msa.length, 21);
    assert.equal(af3MsaFromA3m(text).depth, 22);
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

  it("gives the paired block at most half of AF3's 16384 rows, and the rest to unpaired", () => {
    // 🔴 THE REGRESSION THIS FILE EXISTED WITHOUT. AF3 sets max_paired_sequences = msa_size // 2 and then fills the
    // remainder from the unpaired block. Letting the paired block take the whole crop is the natural way to write the
    // loop, and it only shows on a HOMO-OLIGOMER: a monomer has no paired block, so every single-chain check passes
    // while a dimer folds against a paired block that has eaten everything, no unpaired rows at all, and a profile
    // with nothing left to average.
    const deep = (residue, count) => a3m(QUERY, ...Array.from({ length: count }, () => residue.repeat(10)));
    const rows = af3MsaFromA3m({ paired: deep("A", 9000), unpaired: deep("C", 10000) });
    // The paired block (its A3M's 9001 rows, plus the query row featuriseProtein writes) is cropped to 8192, so
    // 8191 rows come from here. The unpaired crop is the remaining 8192, and it spends 8191 of them on HOMOLOGS: its
    // own query row is the duplicate AF3's deduplication removes.
    assert.equal(rows.depth, AF3_MSA_SIZE - 1);
    assert.equal(rows.unpairedFrom, 8192);
    assert.equal(rows.msa.filter((row) => row[0] === AF3_MSA_CODES.A).length, 8191);
    assert.equal(rows.msa.filter((row) => row[0] === AF3_MSA_CODES.C).length, 8191);
    // ...and the order is file order: every paired row before every unpaired one
    assert.equal(rows.msa[8190][0], AF3_MSA_CODES.A);
    assert.equal(rows.msa[8191][0], AF3_MSA_CODES.C);
  });

  it("gives the unpaired block everything when there is no paired one", () => {
    const deep = a3m(QUERY, ...Array.from({ length: 40 }, () => "CCCCCCCCCC"));
    const rows = af3MsaFromA3m({ paired: null, unpaired: deep });
    assert.equal(rows.depth, 42);
    assert.equal(rows.unpairedFrom, 1, "straight after the query row");
  });

  it("keeps the alignment's own order, and nothing depends on a seed", () => {
    const text = a3m(QUERY, "AAAAAAAAAA", "CCCCCCCCCC", "DDDDDDDDDD");
    const { msa } = af3MsaFromA3m(text);
    assert.deepEqual(msa.map((row) => row[0]), [AF3_MSA_CODES.A, AF3_MSA_CODES.A, AF3_MSA_CODES.C, AF3_MSA_CODES.D]);
  });
});

describe("the pool a deeper alignment gives", () => {
  it("leaves the profile alone, which is over the whole alignment", () => {
    const deep = a3m(QUERY, ...Array.from({ length: 40 },
      (_, index) => "ACDEFGHIK" + "LMNPQRSTVWY"[index % 11]));
    const rows = af3MsaFromA3m({ unpaired: deep });
    assert.equal(rows.profileMsa.length, 41, "the query and all 40 hits");
  });

  it("is AF3's 16384-row MSA, not past it - and the profile is over them all", () => {
    // features.py crops each chain to msa_size rows (less the paired block) before the model draws num_msa; rows
    // past that are never drawn. Rows from 16382 on are W here, so a W in the pool is a row AF3 never holds.
    const rows = ["CCCCCCCCCC"];
    for (let row = 1; row < 20000; row += 1) rows.push(row < AF3_MSA_SIZE - 2 ? "CCCCCCCCCC" : "WWWWWWWWWW");
    rows[0] = QUERY;
    const out = af3MsaFromA3m({ paired: a3m(QUERY), unpaired: a3m(...rows) });
    assert.equal(AF3_MSA_SIZE, 16384);
    assert.ok(out.msa.every((codes) => codes[0] !== AF3_MSA_CODES.W), "a row past AF3's crop is in the pool");
    // (the pool is 16384 less the paired block of two rows: rows 0..16381, so 16381 is in and 16382 out)
    assert.equal(out.msa.filter((codes) => codes[0] === AF3_MSA_CODES.C).length, AF3_MSA_SIZE - 3);
    assert.equal(out.profileMsa.length, 20000, "the profile is over the uncropped block");
  });
});
