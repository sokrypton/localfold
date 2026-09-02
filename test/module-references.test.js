/**
 * Every name a module calls is a name that module can see.
 *
 * 🔴 THIS SHIPPED, AND ONLY THE STOP BUTTON FOUND IT. web/app.js called
 * mergeSearchedChains without importing it, so the MSA-reuse path threw
 * "mergeSearchedChains is not defined" - and that path only runs on a SECOND
 * fold of a complex, after a first search has filled the cache. Stopping a fold
 * partway is one of the few ways to reach it, which is how a user hit it and
 * nothing else did. There is no linter in this repository, so nothing else
 * would have.
 *
 * 🔴 IT IS A HEURISTIC AND SAYS SO. It only considers names some module under
 * src/ EXPORTS - a typo'd local variable is still nobody's business here - and
 * it treats a name as visible if the file imports it, declares it, or exports
 * it. That is enough to catch the class of bug above, which is a name that
 * exists in the project but not in the file using it, and cheap enough to need
 * no dependency.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "./harness.js";

/**
 * 🔴 COMMENTS AND STRINGS ARE STRIPPED FIRST, and without that this is unusable.
 * This repository comments heavily and names functions in prose - "see
 * conditionedTransition" - so a scanner that reads comments reports every one of
 * them as a call. Template literals matter too: the WGSL shaders live in them
 * and are full of `f32(` and `min(`.
 */
function stripCommentsAndStrings(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1 ")
    .replace(/`(?:\\.|[^`\\])*`/g, " ")
    .replace(/'(?:\\.|[^'\\\n])*'/g, " ")
    .replace(/"(?:\\.|[^"\\\n])*"/g, " ");
}

function walk(directory, out = []) {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) walk(path, out);
    else if (entry.endsWith(".js") && !entry.endsWith(".min.js")) out.push(path);
  }
  return out;
}

/** Names introduced by `export function f`, `export const f`, `export class f`. */
function exportedNames(source) {
  const names = new Set();
  for (const match of source.matchAll(/^export\s+(?:async\s+)?(?:function|const|let|class)\s+([A-Za-z_$][\w$]*)/gm)) {
    names.add(match[1]);
  }
  return names;
}

/** Everything the file can see: imports, its own declarations, its own exports. */
function visibleNames(source) {
  const names = new Set();
  for (const match of source.matchAll(/import\s*\{([^}]*)\}/g)) {
    for (const part of match[1].split(",")) {
      const name = part.split(/\s+as\s+/).pop().trim();
      if (name !== "") names.add(name);
    }
  }
  for (const match of source.matchAll(/import\s+([A-Za-z_$][\w$]*)\s*(?:,|from)/g)) {
    names.add(match[1]);
  }
  for (const match of source.matchAll(/(?:^|\s)(?:async\s+)?(?:function|const|let|var|class)\s+([A-Za-z_$][\w$]*)/gm)) {
    names.add(match[1]);
  }
  // 🔴 CLASS METHODS COUNT AS DECLARATIONS. `async tensor(name) {` is a method
  // definition, and without this the scanner reads it as a call to an import
  // that is not there - which is every store in src/reference/.
  for (const match of source.matchAll(/^\s*(?:static\s+)?(?:async\s+)?#?([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*\{/gm)) {
    names.add(match[1]);
  }
  return names;
}

describe("module references", () => {
  it("calls no exported name the file cannot see", () => {
    const sources = new Map();
    for (const path of [...walk("src"), ...walk("web")]) {
      sources.set(path, stripCommentsAndStrings(readFileSync(path, "utf8")));
    }
    const projectExports = new Set();
    for (const [path, source] of sources) {
      if (!path.startsWith("src")) continue;
      for (const name of exportedNames(source)) projectExports.add(name);
    }

    const problems = [];
    for (const [path, source] of sources) {
      const visible = visibleNames(source);
      const own = exportedNames(source);
      const called = new Set();
      for (const match of source.matchAll(/([A-Za-z_$][\w$]*)\s*\(/g)) called.add(match[1]);
      for (const name of called) {
        if (!projectExports.has(name)) continue;
        if (visible.has(name) || own.has(name)) continue;
        // `foo.bar(` is a method call, not a reference to our export.
        const bare = new RegExp(`(^|[^.\\w$])${name}\\s*\\(`, "m");
        if (!bare.test(source)) continue;
        problems.push(`${path} calls ${name} without importing or declaring it`);
      }
    }
    expect(problems).toEqual([]);
  });
});
