import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { sourceFiles } from "./helpers/source-files.js";
import { dirname, join, normalize, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

/**
 * No directory in `src/` may depend on a directory that depends on it back.
 *
 * 🔴 BECAUSE THERE WERE THREE, AND EACH ONE WAS A FILE IN THE WRONG PLACE. A
 * cycle between two directories is not a style complaint: it means neither can
 * be read, moved or reasoned about without the other, and here every one of
 * them turned out to be a single leaf module filed under the wrong heading.
 *
 *   reference <-> runtime   `dtype.js` - "reading a stored tensor", a RUNTIME
 *                           concern sitting in the weight-bundle layer
 *   design <-> af3          `superpose-pdb.js` - imported by af3 and web and
 *                           NOT by design, the directory it lived in
 *   af3 <-> evoformer       the one that is real; see ALLOWED below
 *
 * Both of the first two were leaves - no relative imports of their own - so
 * moving them could not create a new edge, and both cycles closed.
 */
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "src");

/**
 * 🔴 EMPTY, AND IT HAS BEEN EMPTY SINCE THE LAST CYCLE CLOSED. `af3 <->
 * evoformer` was here with its reason - AlphaFold 2's template module using
 * AlphaFold 3's template featuriser, and AlphaFold 3's grid attention using
 * AlphaFold 2's subgroup-matrix geometry - and it was closed by taking the
 * shared surface out to `src/kernels/attention-geometry.js` rather than by
 * moving a file. THIS GATE IS WHAT SAID SO: the entry stopped matching and the
 * second assertion below failed with "af3<->evoformer is allowed and no longer
 * happens - drop it", which is the whole reason that assertion exists.
 *
 * An entry here is a promise to explain, not a way to make the gate quiet.
 */
const ALLOWED = new Set([]);

/** The top-level `src/` directory a file belongs to, or "(root)" for a loose one. */
const areaOf = (path) => {
  const parts = relative(ROOT, path).split(sep);
  return parts.length > 1 ? parts[0] : "(root)";
};

test("src has no import cycles between directories", () => {
  const files = sourceFiles(ROOT);
  assert.ok(files.length >= 150, `walked ${files.length} files`);

  const edges = new Map();
  let specifiers = 0;
  for (const file of files) {
    const from = areaOf(file);
    const text = readFileSync(file, "utf8")
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .split("\n").filter((line) => !/^\s*\/\//.test(line)).join("\n");
    for (const match of text.matchAll(/(?:from|import)\s*\(?\s*["'](\.[^"']+)["']/g)) {
      specifiers += 1;
      const to = areaOf(normalize(join(dirname(file), match[1])));
      if (to === from) continue;
      if (!edges.has(from)) edges.set(from, new Set());
      edges.get(from).add(to);
    }
  }
  // 🔴 A RULE THAT STOPS MATCHING PASSES BY FINDING NOTHING.
  assert.ok(specifiers >= 300, `only ${specifiers} relative imports seen`);

  const cycles = [];
  for (const [from, targets] of edges) {
    for (const to of targets) {
      if (!edges.get(to)?.has(from)) continue;
      const name = [from, to].sort().join("<->");
      if (!ALLOWED.has(name) && !cycles.includes(name)) cycles.push(name);
    }
  }
  assert.deepEqual(cycles.sort(), [], `${cycles.length} unexplained directory cycles`);

  // ...and an allowance that has been fixed must be REMOVED, or the next one
  // gets added beside a stale entry and nobody notices.
  const live = [];
  for (const [from, targets] of edges) {
    for (const to of targets) {
      if (edges.get(to)?.has(from)) live.push([from, to].sort().join("<->"));
    }
  }
  for (const allowed of ALLOWED) {
    assert.ok(live.includes(allowed), `${allowed} is allowed and no longer happens - drop it`);
  }
});
