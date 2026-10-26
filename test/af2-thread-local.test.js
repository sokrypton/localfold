/**
 * Every mutable global and function-level static AlphaFold 2's native port reads is its thread's own (LF_TLS).
 *
 * 🔴 `localfold-af2 --multi` RUNS SEVERAL MODELS' SERVERS ON THREADS OF ONE PROCESS, and that is only sound if no
 * thread can see another's state: cuda/af2/src/af2.cu defines LF_TLS as thread_local and every mutable variable on
 * its path is declared with it (cuda/af2/README.md, "AlphaFold 2's five models at once"). The conversion was
 * mechanical and missed two kinds of declaration that a text match takes for something else - fourteen weight and
 * tile caches whose `const` sat inside their template arguments (`std::map<const float*, half*>` read as a constant),
 * and FORGET_HOOKS, whose `std::function<void()>` read as a function - so five threads' warm-ups wrote one shared
 * std::map and one shared vector, and every fold still came back byte-identical. A race that passes is still a race;
 * this is the rule, walked over every file af2.cu includes.
 *
 * CAPTURE_MU is process-wide ON PURPOSE: a graph capture on one thread and a device-wide sync on another must exclude
 * each other. Constants, device symbols and functions are not state.
 */
import { strict as assert } from "node:assert";
import { it } from "node:test";
import { readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
function reachable(entry) {
  const out = new Set(), todo = [entry];
  while (todo.length) {
    const file = todo.pop();
    if (out.has(file)) continue;
    out.add(file);
    for (const m of readFileSync(file, "utf8").matchAll(/^\s*#include\s+"([^"]+)"/gm)) todo.push(join(dirname(file), m[1]));
  }
  return [...out];
}
const PROCESS_WIDE = new Set(["CAPTURE_MU"]);
// the declaration's type and name, with template arguments taken out (their `const` and `()` are not the variable's)
const flatten = (text) => { let t = text, prev; do { prev = t; t = t.replace(/<[^<>]*>/g, "<>"); } while (t !== prev); return t; };

it("every mutable variable on cuda/af2's path is LF_TLS (one model a thread, under --multi)", () => {
  const files = reachable(join(ROOT, "cuda/af2/src/af2.cu")).filter((f) => !f.includes("/featurise/"));
  assert.ok(files.length >= 15, `the include walk found only ${files.length} files`);
  const offenders = [];
  let checked = 0;
  for (const file of files) {
    readFileSync(file, "utf8").split("\n").forEach((line, i) => {
      // file scope: `inline X name ...` / `static X name ...` at column 0; function scope: an indented `static`
      const m = line.match(/^(\s*)(inline|static)\s+(.*)$/);
      if (!m || (m[2] === "inline" && m[1])) return;
      const rest = flatten(m[3]);
      const cut = rest.search(/[=;{[]/);
      if (cut < 0) return;
      const head = rest.slice(0, cut);
      if (head.includes("(")) return;                                        // a function (or its declaration)
      if (/\b(constexpr|const|__device__|__shared__|__constant__|extern|inline|__global__)\b/.test(head)) return;
      if (/^\s*(struct|class|enum|union)\b/.test(head) && !/\b\w+\s*$/.test(head.replace(/\{.*/, ""))) return;
      const name = (head.match(/(\w+)\s*$/) || [])[1];
      if (!name || /^(void|int|bool|float|double)$/.test(name)) return;
      checked++;
      if (/\b(LF_TLS|thread_local)\b/.test(head) || PROCESS_WIDE.has(name)) return;
      offenders.push(`${relative(ROOT, file)}:${i + 1} ${line.trim().slice(0, 100)}`);
    });
  }
  assert.ok(checked >= 120, `checked only ${checked} declarations - the pattern stopped matching`);
  assert.deepEqual(offenders, [], "shared state a --multi thread would see from another (declare it LF_TLS)");
});
