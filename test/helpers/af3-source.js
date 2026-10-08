import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { sourceFiles } from "./source-files.js";

/**
 * Reading AF3 source BY NAME, wherever in its tree's af3/ it lives.
 *
 * 🔴 BECAUSE THREE STRUCTURAL TESTS OPENED `src/af3/<name>.js` BY PATH AND THE
 * REORGANISATION BROKE ALL THREE AT ONCE. `test/imports-resolve.test.js` cannot
 * see them: they are `readFileSync` calls, not imports, so the one gate written
 * to make a file move safe was blind to exactly the tests that inspect files.
 *
 * A test that asserts something about the TEXT of a module should name the
 * module, not its directory - the directory is a fact about how the tree is
 * organised today and every one of these tests is about something else.
 */
// AF3's sources are in three trees - shared/af3, cpu/af3 and webgpu/af3 - and a
// stage's reference and its kernel share a basename across two of them
// (cpu/af3/trunk/embedder.js beside webgpu/af3/trunk/embedder.js), so a source
// is named by its TREE and its basename: "webgpu:embedder.js". The tree is the
// backend, which is what such a test is about; the stage directory is not.
const TREES = ["shared", "cpu", "webgpu"];

function byName() {
  const out = new Map();
  for (const tree of TREES) for (const path of sourceFiles(new URL(`../../${tree}/af3/`, import.meta.url).pathname)) {
    const name = `${tree}:${basename(path)}`;
    // 🔴 A DUPLICATE BASENAME WOULD MAKE THIS AMBIGUOUS AND SILENT, so it
    // raises instead: two `template-features.js` under one tree's af3 and a caller
    // gets whichever the walk reached last.
    if (out.has(name)) {
      throw new Error(`two files named ${name} under shared/, cpu/ and webgpu/af3: ${out.get(name)} and ${path}`);
    }
    out.set(name, path);
  }
  return out;
}

const BY_NAME = byName();

/** Every AF3 source, "tree:basename" -> text. */
export function af3Sources() {
  return new Map([...BY_NAME].map(([name, path]) => [name, readFileSync(path, "utf8")]));
}

/** One AF3 source by "tree:basename", raising rather than returning undefined. */
export function af3Source(name) {
  const path = BY_NAME.get(name);
  if (path === undefined) {
    throw new Error(`no ${name} under shared/, cpu/ or webgpu/af3 (have ${BY_NAME.size} files)`);
  }
  return readFileSync(path, "utf8");
}
