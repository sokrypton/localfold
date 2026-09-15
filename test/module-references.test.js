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
  // 🔴 A SCANNER, NOT FIVE REGEXES, BECAUSE A NESTED TEMPLATE LITERAL HID A
  // LIVE BUG. The regex version paired backticks strictly in order, so one
  // template inside another - `${ok ? `a` : `b`}` - put the rest of the file
  // out of step: stretches of real CODE read as string content and were blanked.
  // web/app.js had such a template at line 3172, and at line 3418 it called
  // `chainGeometryVerdict` without importing it. The regex stripper removed the
  // call, this check passed, and every AlphaFold 2 fold on the live page ended in
  // "chainGeometryVerdict is not defined". It hid real calls in
  // src/af3/trunk/trunk-webgpu.js too.
  //
  // What this keeps that the regexes did not: CODE INSIDE `${...}`. A template's
  // literal text is prose or WGSL and is blanked; its expressions are JavaScript
  // and can call an unimported name like any other line.
  const chars = source.split("");
  const n = source.length;
  const blank = (from, to) => {
    for (let k = from; k < Math.min(to, n); k += 1) if (chars[k] !== "\n") chars[k] = " ";
  };
  const templates = []; // one brace depth per open `${`
  let i = 0;
  // A `/` starts a regex literal after an operator, an opening bracket or a
  // keyword, and is division after a value. Getting it wrong either way lets a
  // quote or backtick inside a regex open a phantom string.
  const regexCanStart = () => {
    let j = i - 1;
    while (j >= 0 && (chars[j] === " " || chars[j] === "\t" || chars[j] === "\n" || chars[j] === "\r")) j -= 1;
    if (j < 0) return true;
    if ("(,=:[!&|?{};+-*%<>~^".includes(source[j])) return true;
    const word = source.slice(Math.max(0, j - 10), j + 1).match(/[A-Za-z_$]+$/)?.[0];
    return ["return", "typeof", "case", "in", "of", "void", "delete", "throw"].includes(word);
  };
  // Scan a template's literal text from `i`, blanking from `start`, up to its
  // closing backtick or its next `${`.
  const readTemplate = (start) => {
    while (i < n) {
      if (source[i] === "\\") { i += 2; continue; }
      if (source[i] === "`") { blank(start, i + 1); i += 1; return; }
      if (source[i] === "$" && source[i + 1] === "{") { blank(start, i); i += 2; templates.push(0); return; }
      i += 1;
    }
    blank(start, n);
  };
  while (i < n) {
    const c = source[i];
    const d = source[i + 1];
    if (c === "/" && d === "/") {
      const newline = source.indexOf("\n", i);
      const stop = newline === -1 ? n : newline;
      blank(i, stop); i = stop; continue;
    }
    if (c === "/" && d === "*") {
      const close = source.indexOf("*/", i + 2);
      const stop = close === -1 ? n : close + 2;
      blank(i, stop); i = stop; continue;
    }
    if (c === "'" || c === "\"") {
      const begin = i; i += 1;
      while (i < n && source[i] !== c && source[i] !== "\n") i += source[i] === "\\" ? 2 : 1;
      blank(begin, i + 1); i += 1; continue;
    }
    if (c === "`") { const begin = i; i += 1; readTemplate(begin); continue; }
    if (c === "/" && regexCanStart()) {
      const begin = i; i += 1;
      let inClass = false;
      while (i < n && source[i] !== "\n") {
        const ch = source[i];
        if (ch === "\\") { i += 2; continue; }
        if (ch === "[") inClass = true;
        else if (ch === "]") inClass = false;
        else if (ch === "/" && !inClass) break;
        i += 1;
      }
      blank(begin, i + 1); i += 1; continue;
    }
    if (templates.length > 0) {
      if (c === "{") templates[templates.length - 1] += 1;
      else if (c === "}") {
        if (templates[templates.length - 1] === 0) { templates.pop(); i += 1; readTemplate(i); continue; }
        templates[templates.length - 1] -= 1;
      }
    }
    i += 1;
  }
  return chars.join("");
}

/**
 * 🔴 THE VENDORED MPNN MIRROR IS NOT THIS REPOSITORY'S CODE AND IS SKIPPED.
 * Two reasons, and the second is the one that matters. It reports a false
 * positive - `accel.js` reaches `useAccelerator` through
 * `const { useAccelerator } = await import("./ops.js")`, which this scanner's
 * import patterns do not model - and there is nothing to be done about it
 * here, because src/design/mpnn/ is a MIRROR: editing it is reverted by the
 * next `python3 tools/sync-mpnn.py`. And its exports would otherwise join
 * `projectExports`, so common names it ships (`linear`, `softmax`,
 * `layerNorm`) would start being looked for in every unrelated file.
 * test/mpnn-vendored.test.js is what holds that directory to anything.
 */
const VENDORED = join("src", "design", "mpnn");

function walk(directory, out = []) {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (path === VENDORED) continue;
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
  // that is not there - which is every store in src/bundles/.
  for (const match of source.matchAll(/^\s*(?:static\s+)?(?:async\s+)?#?([A-Za-z_$][\w$]*)\s*\([^()]*\)\s*\{/gm)) {
    names.add(match[1]);
  }
  return names;
}

describe("the comment and string stripper", () => {
  const calls = (source, name) =>
    new RegExp(`(^|[^.\\w$])${name}\\s*\\(`, "m").test(stripCommentsAndStrings(source));

  // 🔴 THE SHAPE OF THE BUG THAT SHIPPED: a nested template, then a call.
  it("still sees a call after a template literal nested inside another", () => {
    const source = [
      "const label = `outer ${ready ? `inner` : \"x\"} tail`;",
      "const chain = chainGeometryVerdict(result.geometry);",
    ].join("\n");
    expect(calls(source, "chainGeometryVerdict")).toBe(true);
  });

  it("keeps code inside ${...} and blanks the template's prose", () => {
    const source = "const text = `see describeThing( in prose ${formatValue(1)} and more`;";
    expect(calls(source, "formatValue")).toBe(true);
    expect(calls(source, "describeThing")).toBe(false);
  });

  it("blanks comments, strings and regex literals that look like calls", () => {
    const source = [
      "// commentCall(1)",
      "/* blockCall(2) */",
      "const a = 'quotedCall(3)';",
      "const b = /regexCall\\(4\\)'`/;",
      "realCall(5);",
    ].join("\n");
    for (const name of ["commentCall", "blockCall", "quotedCall", "regexCall"]) {
      expect(calls(source, name)).toBe(false);
    }
    expect(calls(source, "realCall")).toBe(true);
  });
});

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
