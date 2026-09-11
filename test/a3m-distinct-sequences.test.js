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

import { distinctSequenceCount, foundOnlyTheQuery, parseA3m } from "../src/input/a3m.js";

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
