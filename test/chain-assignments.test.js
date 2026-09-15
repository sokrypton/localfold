import test from "node:test";
import assert from "node:assert/strict";
import { chainAssignments } from "../tools/gpu/superpose.js";

const key = (list) => list.map((a) => a.join("")).sort();

test("chainAssignments", async (t) => {
  await t.test("a heterodimer has exactly one labelling", () => {
    assert.deepEqual(chainAssignments(["AAA", "CCC"]), [[0, 1]]);
  });

  // 🔴 THE ONE THAT MATTERS: 5CAJ is the same sequence twice.
  await t.test("a homodimer has two", () => {
    assert.deepEqual(key(chainAssignments(["AAA", "AAA"])), ["01", "10"]);
  });

  await t.test("the identity comes first, so an unpermuted score is reportable", () => {
    assert.deepEqual(chainAssignments(["AAA", "AAA"])[0], [0, 1]);
    assert.deepEqual(chainAssignments(["AAA", "AAA", "AAA"])[0], [0, 1, 2]);
  });

  await t.test("a homotrimer has six", () => {
    assert.equal(chainAssignments(["A", "A", "A"]).length, 6);
  });

  // 🔴 AND A MIXED COMPLEX PERMUTES WITHIN ITS GROUPS ONLY. Swapping the two
  // copies of one protein is a relabelling; swapping a copy with the other
  // protein is a different answer.
  await t.test("two copies of one chain beside a different one gives two", () => {
    const got = key(chainAssignments(["AAA", "CCC", "AAA"]));
    assert.deepEqual(got, ["012", "210"]);
  });

  await t.test("refuses more than it will enumerate", () => {
    assert.throws(() => chainAssignments(Array(8).fill("A")), /refuses above 720/);
    assert.equal(chainAssignments(Array(6).fill("A")).length, 720);
    assert.throws(() => chainAssignments(["A", "A", "A"], { limit: 2 }), /refuses above 2/);
  });
});
