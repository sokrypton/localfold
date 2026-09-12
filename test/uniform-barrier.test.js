/**
 * No barrier sits inside a loop whose trip count differs between lanes.
 *
 * 🔴 THIS IS THE "RACE" IN AF2's MATRIX FLASH ATTENTION, WHICH WAS NOT ONE. A
 * staging loop written `for (var i = local; i < COUNT; i += LANES)` runs a
 * different number of times per lane whenever `COUNT` is not a multiple of
 * `LANES` - and an AF2 fold compiles that kernel at a head of EIGHT, where the
 * count is 64 against 128 lanes, so the second half of the workgroup never
 * enters the loop at all. Two things follow, and both were done to it during a
 * bisection and reported as evidence of a race in the shipped kernel:
 *
 *   - a `workgroupBarrier()` inside such a loop is in non-uniform control flow,
 *     which WGSL leaves undefined;
 *   - unrolling it to a fixed number of iterations without keeping the `i <
 *     COUNT` guard writes past the arrays it stages into.
 *
 * Both reproduce as a fold that differs run to run, which is what
 * `bench-af2-warm.js` reports and why the conclusion was "a pure reordering
 * cannot make a correct kernel wrong, so it was already wrong". The reordering
 * was not pure. See docs/A100.md.
 *
 * The rule here covers the FIRST of the two, because it is the one a reader
 * cannot see: an unguarded unroll is visible in the diff and a barrier inside a
 * lane-strided loop looks like every other barrier in the file. There are no
 * instances today; this is what keeps it that way.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

// A loop whose induction variable STARTS at a lane identifier. That is the
// strided-over-the-workgroup idiom, and the only one whose trip count can
// differ between the lanes of one workgroup.
const LOOP = /for\s*\(\s*var\s+(\w+)\s*(?::\s*u32\s*)?=\s*([^;]*?)\s*;/g;
const LANE = /\blocal\b|\blocal_id\b|\blane\b|\bid\.x\b|local_invocation/;
const BARRIER = /workgroupBarrier|storageBarrier/;

// 🔴 A FLOOR, BECAUSE A RULE THAT STOPS MATCHING PASSES BY FINDING NOTHING.
// 189 such loops across 28 files when this was written; a regexp that quietly
// covers a tenth of them would report zero violations forever.
const AT_LEAST = 150;

function* sources(directory) {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) { yield* sources(path); continue; }
    if (path.endsWith(".js")) yield path;
  }
}

/**
 * The text between a loop header's `{` and its matching `}`.
 *
 * 🔴 NOT `indexOf("{")`. These shaders are JavaScript template literals and a
 * loop header reads `i += ${LANES}u) {`, so the first brace after the header is
 * the one opening the INTERPOLATION - which made the whole rule scan the string
 * "{LANES}" and report no violations at all. Caught by injecting a barrier into
 * a real loop and watching the test pass; the arm below is that, kept.
 */
function body(text, from) {
  let open = from;
  while (true) {
    open = text.indexOf("{", open);
    if (open < 0) return null;
    if (text[open - 1] !== "$") break;
    open += 1;
  }
  let depth = 0;
  for (let at = open; at < text.length; at += 1) {
    if (text[at] === "{") depth += 1;
    else if (text[at] === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(open, at);
    }
  }
  return null;
}

export function laneStridedLoops(text) {
  const found = [];
  for (const match of text.matchAll(LOOP)) {
    if (!LANE.test(match[2])) continue;
    const inner = body(text, match.index + match[0].length);
    found.push({ header: match[0].trim(), line: text.slice(0, match.index).split("\n").length,
                 read: inner !== null && inner.length > 40,
                 barrier: inner !== null && BARRIER.test(inner) });
  }
  return found;
}

describe("a barrier is never inside a lane-strided loop", () => {
  const scanned = [...sources("src")].map((path) => ({
    path, loops: laneStridedLoops(readFileSync(path, "utf8")),
  }));
  const loops = scanned.reduce((sum, file) => sum + file.loops.length, 0);

  it("still finds the loops it is meant to be checking", () => {
    assert.ok(loops >= AT_LEAST,
      `matched ${loops} lane-strided loops, expected at least ${AT_LEAST}: `
      + "the pattern has gone stale and this rule is checking nothing");
  });

  // 🔴 AND THAT IT READ THEIR BODIES, which is the failure that actually
  // happened: the brace matcher returned "{LANES}" for every loop whose header
  // interpolates, so the rule matched 189 loops and inspected none of them.
  // Counting matches is not enough - a rule can be vacuous one level down.
  it("reads a real body for substantially all of them", () => {
    const read = scanned.reduce(
      (sum, file) => sum + file.loops.filter((loop) => loop.read).length, 0);
    assert.ok(read >= loops * 0.8,
      `extracted a body for only ${read} of ${loops} loops: the brace matcher `
      + "is landing on an interpolation rather than the loop");
  });

  it("has no barrier in any of them", () => {
    const bad = scanned.flatMap(({ path, loops: found }) => found
      .filter((loop) => loop.barrier)
      .map((loop) => `${path}:${loop.line} ${loop.header}`));
    assert.deepEqual(bad, [],
      `a barrier inside a loop whose trip count can differ between lanes is `
      + `undefined behaviour in WGSL:\n  ${bad.join("\n  ")}`);
  });

  it("fails on the shape it exists to catch", () => {
    const bad = laneStridedLoops(`
      for (var i = local; i < KEYS * HD4; i += 128u) {
        staged[i] = read(i);
        workgroupBarrier();
      }`);
    assert.equal(bad.length, 1);
    assert.equal(bad[0].barrier, true);
    // ...and a loop over a constant bound is not the shape, however many
    // barriers it holds: every lane runs it the same number of times.
    assert.deepEqual(laneStridedLoops(`
      for (var t = 0u; t < 4u; t += 1u) { workgroupBarrier(); }`), []);
  });
});
