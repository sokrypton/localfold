// The per-pass MSA draw (shared/af3/msa-draw.js), and the values its C++ twin (cuda/featurise/msa_draw.h) must give:
// cuda/af3 and metal/af3 draw with the twin, the page with this, and a seed has to name the same rows on all three.
import { test } from "node:test";
import assert from "node:assert/strict";
import { msaDraw, msaDrawRows, gatherRows, packMsa, msaNativeRows, mcDropoutApplies, mcDropoutKey, mcDropoutThreshold,
  mcDropoutKeep } from "../shared/af3/msa-draw.js";

test("a draw is `count` distinct rows of the pool, ascending", () => {
  const rows = msaDraw(42, 3, 5630, 1024);
  assert.equal(rows.length, 1024);
  for (let k = 1; k < rows.length; k += 1) assert.ok(rows[k] > rows[k - 1]);
  assert.ok(rows[0] >= 0 && rows[1023] < 5630);
});

test("each pass draws afresh, and a seed and pass name one draw", () => {
  const a = msaDraw(7, 0, 5630, 1024), b = msaDraw(7, 1, 5630, 1024);
  assert.notDeepEqual(a, b);
  assert.deepEqual(msaDraw(7, 1, 5630, 1024), b);
  assert.notDeepEqual(msaDraw(8, 1, 5630, 1024), b);
  // and across eleven passes most of the pool is seen - AF3's point, and what one draw could not do (8S83)
  const seen = new Set();
  for (let pass = 0; pass < 11; pass += 1) for (const row of msaDraw(7, pass, 5630, 1024)) seen.add(row);
  assert.ok(seen.size > 4500, `${seen.size} rows seen in eleven passes`);
});

test("the values cuda/featurise/msa_draw.h gives (checked against the C++ when written)", () => {
  // [seed, pass, rows[0], rows[1], rows[1023], sum of rows[i] * (i + 1)] at 5630 rows and 1024 drawn
  const golden = [
    [0, 0, 4, 9, 5623, 1966111906], [0, 1, 3, 11, 5627, 1929997402], [42, 2, 2, 8, 5625, 1952192954],
    [4294967297, 0, 6, 8, 5627, 1956745732], [9007199254740991, 2, 0, 5, 5629, 1949733050],
  ];
  for (const [seed, pass, first, second, last, sum] of golden) {
    const rows = msaDraw(seed, pass, 5630, 1024);
    let weighted = 0;
    rows.forEach((row, at) => { weighted += row * (at + 1); });
    assert.deepEqual([rows[0], rows[1], rows[1023], weighted], [first, second, last, sum], `seed ${seed} pass ${pass}`);
  }
  assert.deepEqual(Array.from(msaDraw(7, 3, 10, 4)), [4, 5, 7, 9]);
});

test("the modes: once repeats pass 0, prefix takes the first rows, a shallow pool is every row", () => {
  assert.deepEqual(msaDrawRows("once", 5, 9, 100, 10), msaDraw(5, 0, 100, 10));
  assert.deepEqual(msaDrawRows("pass", 5, 9, 100, 10), msaDraw(5, 9, 100, 10));
  assert.deepEqual(Array.from(msaDrawRows("prefix", 5, 9, 100, 4)), [0, 1, 2, 3]);
  assert.deepEqual(Array.from(msaDrawRows("pass", 5, 9, 3, 10)), [0, 1, 2]);
  assert.throws(() => msaDrawRows("shuffle", 5, 0, 100, 10), /not pass, once, prefix or native/);
});

test("gatherRows takes whole rows, in the order asked", () => {
  const pool = Int32Array.from({ length: 12 }, (_, at) => at);      // 4 rows of 3
  assert.deepEqual(Array.from(gatherRows(pool, 3, [1, 3])), [3, 4, 5, 9, 10, 11]);
  assert.ok(gatherRows(new Float32Array(6), 3, [0]) instanceof Float32Array);
});

test("packMsa: an int32 a cell, code | deletion << 8 | mask << 16, as the native trunks unpack it", () => {
  const packed = packMsa(Int8Array.of(21, 3, 31), Uint8Array.of(1, 0, 1), Uint8Array.of(0, 127, 5));
  assert.deepEqual(Array.from(packed), [21 | (1 << 16), 3 | (127 << 8), 31 | (5 << 8) | (1 << 16)]);
  // (the unpacking msa_draw.h's unpackMsaRow does)
  const v = packed[1];
  assert.deepEqual([v & 0xff, (v >> 8) & 0xff, (v >> 16) & 1], [3, 127, 0]);
});

test("protenix's native draw: a count from 1 to all, uncapped, as msa_draw.h draws it", () => {
  // [seed, pass, count, first row, sum of rows[i] * (i + 1), the pass's dropout key] at 5631 rows (checked against the C++)
  const golden = [[0, 0, 5068, 0, 48210686772, 320700260], [1, 2, 1218, 0, 2788324409, 1342873464],
                  [4294967297, 1, 932, 5, 1637193110, 3316591503]];
  for (const [seed, pass, count, first, sum, key] of golden) {
    const rows = msaNativeRows(seed, pass, 5631);
    let weighted = 0;
    rows.forEach((row, at) => { weighted += row * (at + 1); });
    assert.deepEqual([rows.length, rows[0], weighted, mcDropoutKey(seed, pass)], [count, first, sum, key]);
  }
  assert.deepEqual(msaDrawRows("native", 1, 2, 5631, 1024), msaNativeRows(1, 2, 5631));
  // a single row is always itself, and the counts span the pool
  assert.deepEqual(Array.from(msaNativeRows(9, 4, 1)), [0]);
  const counts = Array.from({ length: 200 }, (_, pass) => msaNativeRows(3, pass, 100).length);
  assert.ok(Math.min(...counts) <= 5 && Math.max(...counts) >= 95, `${Math.min(...counts)}..${Math.max(...counts)}`);
});

test("protenix's MC dropout: decided once a seed at its odds, and a pass's mask keeps 1 - rate", () => {
  assert.deepEqual([0, 1, 42, 4294967297].map((seed) => mcDropoutApplies(seed, 0.4)), [false, true, false, true]);
  assert.equal(mcDropoutApplies(1, 0), false);
  const seeds = Array.from({ length: 2000 }, (_, seed) => mcDropoutApplies(seed, 0.4)).filter(Boolean).length;
  assert.ok(Math.abs(seeds / 2000 - 0.4) < 0.04, `${seeds} of 2000`);
  const key = mcDropoutKey(7, 2), threshold = mcDropoutThreshold(0.4);
  let kept = 0;
  for (let row = 0; row < 1000; row += 1) for (let channel = 0; channel < 128; channel += 1) kept += mcDropoutKeep(key, row, channel, threshold);
  assert.equal(kept, 76757);        // (the C++'s count, the same hash)
});
