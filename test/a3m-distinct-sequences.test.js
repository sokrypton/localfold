/**
 * A search that finds nothing but the query is a single-sequence fold.
 *
 * 🔴 `depth` COUNTS ROWS AND IS NOT THE ANSWER. `extractMmseqs2A3m` joins the
 * uniref block and the environmental block and returns each WHOLE, so both
 * begin with their own `>101` - a query with no homologs comes back as **depth
 * 2 carrying one sequence**. Measured on the 59-mer: folding the query twice
 * instead of once moves pTM 0.3965 -> 0.4148 and pLDDT 59.974 -> 60.079, so
 * this is not a cosmetic count.
 *
 * The decision lives here rather than in web/app.js because node cannot import
 * that file - it wants a DOM - and an untested branch in the fold path is the
 * fault docs/WEB.md keeps recording.
 */
import test from "node:test";
import assert from "node:assert/strict";

import { distinctSequenceCount, foldsAsSingleSequence, foundOnlyTheQuery, parseA3m }
  from "../src/input/a3m.js";

const QUERY = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ";
const HIT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVA";

const a3m = (...rows) => `${rows.map((row, index) => `>${index === 0 ? "101" : `hit${index}`}\n${row}`).join("\n")}\n`;

test("a query with no homologs is one sequence however many rows it has", () => {
  // What the server returns with only uniref, and with the environmental
  // database as well - which is the default.
  assert.equal(distinctSequenceCount(a3m(QUERY)), 1);
  assert.equal(distinctSequenceCount(a3m(QUERY, QUERY)), 1);
  assert.equal(parseA3m(a3m(QUERY, QUERY)).depth, 2, "depth still counts rows");
  assert.ok(foundOnlyTheQuery(a3m(QUERY)));
  assert.ok(foundOnlyTheQuery(a3m(QUERY, QUERY)));
});

test("one real homolog is not a single-sequence fold", () => {
  assert.equal(distinctSequenceCount(a3m(QUERY, HIT)), 2);
  assert.equal(distinctSequenceCount(a3m(QUERY, QUERY, HIT)), 2);
  assert.ok(!foundOnlyTheQuery(a3m(QUERY, HIT)));
  // 🔴 AND THE DUPLICATED QUERY MUST NOT HIDE THE ONE HIT. A count that looked
  // at `depth > 2` rather than at distinct rows would call this two homologs
  // and this one none, both wrong.
  assert.ok(!foundOnlyTheQuery(a3m(QUERY, QUERY, HIT)));
});

test("the comparison is on aligned columns, not the raw row", () => {
  // Lowercase columns are insertions; parseA3m drops them into the deletion
  // matrix, so these two rows are the same row to the model - which is what
  // AlphaFold hashes. Comparing raw text would keep a duplicate it removes.
  const withInsertion = `>101\n${QUERY}\n>hit1\n${QUERY.slice(0, 5)}gg${QUERY.slice(5)}\n`;
  assert.equal(parseA3m(withInsertion).depth, 2);
  assert.equal(distinctSequenceCount(withInsertion), 1);
  assert.ok(foundOnlyTheQuery(withInsertion));
});

test("a gapped homolog is a distinct sequence", () => {
  const gapped = `${HIT.slice(0, 10)}-${HIT.slice(11)}`;
  assert.equal(distinctSequenceCount(a3m(QUERY, gapped)), 2);
  assert.ok(!foundOnlyTheQuery(a3m(QUERY, gapped)));
});

test("a pasted alignment folds as a single sequence only when it is OUR sequence", () => {
  // 🔴 AN A3M's OWN QUERY WINS OVER THE SEQUENCE BOX, deliberately, so routing
  // a one-sequence alignment to the query-only path would fold a different
  // protein than the one the reader pasted. Paste-and-upload are the two paths
  // where the two can differ; a searched alignment cannot.
  const other = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTT";
  assert.ok(foldsAsSingleSequence(a3m(QUERY), QUERY));
  assert.ok(foldsAsSingleSequence(a3m(QUERY, QUERY), QUERY));
  // Same alignment, a different protein in the box: keep the alignment.
  assert.ok(!foldsAsSingleSequence(a3m(QUERY), other));
  assert.ok(!foldsAsSingleSequence(a3m(QUERY, QUERY), other));
  // And a real homolog is never a single-sequence fold, matching query or not.
  assert.ok(!foldsAsSingleSequence(a3m(QUERY, HIT), QUERY));
});

test("a complex's concatenated query is what has to match", () => {
  const a = "MKTAYIAKQRQISFVKSHFSRQ";
  const b = "GWSTELEKHREELKEFLKKEGI";
  const merged = a3m(a + b, a + b);
  assert.ok(foldsAsSingleSequence(merged, a + b));
  assert.ok(!foldsAsSingleSequence(merged, a));
});
