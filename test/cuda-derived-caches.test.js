/**
 * Every cache of device weights in the native ports is forgotten with the others.
 *
 * 🔴 THIS WAS A WRONG ESMFold2 FOLD THAT ONLY A COLD PAGE CACHE COULD SHOW. cuda/ef2 warms up WHILE its weights are
 * still arriving (M.uploadAsync), then calls forgetDerivedWeights() so that everything built from the half-uploaded
 * copy - f16 mirrors, packed operands - is rebuilt from the finished one. `triInGemmWeights` kept its packs in a
 * function-local map nobody told to forget, so when the upload was slow (the 6 GB ESM-C 6B tower read from disk
 * rather than from the page cache) the incoming triangle multiplication ran on weights packed from a copy still
 * arriving, for the rest of the process: barnase-barstar 13.2 A and pLDDT 28.6 from a cold cache, 0.58 A and 94.1
 * from a warm one, same binary, same input, same seed. Every gate here runs warm, so none could see it.
 *
 * So the rule is structural: a map holding device pointers, in any file cuda/ef2 includes (its #include graph, walked
 * from src/ef2.cu - the one port that warms up during an upload and so the one that calls forgetDerivedWeights),
 * registers a FORGET_HOOKS entry within the few lines after its declaration - or is one of the named caches
 * forgetDerivedWeights() clears itself, or is not a weight at all. cuda/af3 and cuda/af2 upload their weights
 * before anything runs, so their own caches are out of scope.
 */
import { strict as assert } from "node:assert";
import { it } from "node:test";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
// every file cuda/ef2/src/ef2.cu includes, transitively
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
// file-scope maps forgetDerivedWeights clears by name, or that hold no weight
const NOT_A_HOOK = new Map([
  ["WF", "cleared by forgetDerivedWeights"], ["WH", "cleared by forgetDerivedWeights"],
  ["WH_AT", "cleared by forgetDerivedWeights"], ["BIAS_HALF", "BIAS_HALF_HOOK, beside it"],
  ["SCRATCH", "scratch, rewritten every use, not a weight"], ["IDEV", "the input's own device views"],
  ["AS_BF16", "cuda/af3's pairformer bench, not a port"], ["W", "cuda/af3's pairformer bench, not a port"],
]);
const MAP = /^\s*(?:inline|static)\s+std::(?:unordered_)?map<(.*)>\s+(\w+)\s*;/;

it("every device-weight cache registers a forget hook", () => {
  const offenders = [];
  let seen = 0;
  const files = reachable(join(ROOT, "cuda/ef2/src/ef2.cu"));
  assert.ok(files.length >= 15, `the include walk found only ${files.length} files`);
  for (const file of files) {
    const lines = readFileSync(file, "utf8").split("\n");
    lines.forEach((line, i) => {
      const m = line.match(MAP);
      if (!m || !/\*\s*>?\s*$/.test(m[1].trim()) && !/\*\s*>/.test(m[1])) return;     // the VALUE holds a pointer
      seen++;
      if (NOT_A_HOOK.has(m[2])) return;
      const after = lines.slice(i, i + 4).join("\n");
      if (!after.includes("FORGET_HOOKS")) offenders.push(`${relative(ROOT, file)}:${i + 1} ${m[2]}`);
    });
  }
  assert.ok(seen >= 6, `found only ${seen} device-pointer maps - the pattern stopped matching`);
  assert.deepEqual(offenders, [], "a cache of device weights that forgetDerivedWeights() would not forget");
});
