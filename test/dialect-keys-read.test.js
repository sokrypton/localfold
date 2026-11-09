// Every `dialect.<key>` the AF3-lineage backends read is a key some dialect in shared/af3/dialect.js has.
//
// 🔴 A MISSPELT KEY READS `undefined`, `=== true` MAKES IT `false`, AND NOTHING RAISES. webgl/af3/model/fold.js read
// `dialect.outerProductMeanBiasAfterNorm` where the dialect says `opmBiasAfterNorm`, so boltz2's and rf3's outer product
// mean added its bias on the wrong side of the division on WebGL2 - 2-4e-2 at `z_after_msa` against WebGPU with an
// alignment, ~1e-3 for the five families without the convention, and invisible without one (depth 1, where the two
// orders agree). Comments are skipped; `dialect.js` itself is a file name, not a key.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import * as dialects from "../shared/af3/dialect.js";

const KEYS = new Set(Object.values(dialects)
  .filter((value) => value !== null && typeof value === "object" && "mainMsaCap" in value)
  .flatMap((value) => Object.keys(value)));

const walk = (dir) => readdirSync(dir, { withFileTypes: true }).flatMap((entry) => (entry.isDirectory()
  ? walk(join(dir, entry.name)) : entry.name.endsWith(".js") ? [join(dir, entry.name)] : []));

test("every dialect key webgl/af3 and webgpu/af3 read exists", () => {
  assert.ok(KEYS.size > 50, `found only ${KEYS.size} dialect keys - the dialect filter stopped matching`);
  const unknown = [];
  let reads = 0;
  for (const file of [...walk("webgl/af3"), ...walk("webgpu/af3")]) {
    const code = readFileSync(file, "utf8").replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
    for (const match of code.matchAll(/\bdialect\??\.(\w+)/g)) {
      if (match[1] === "js") continue;
      reads += 1;
      if (!KEYS.has(match[1])) unknown.push(`${file}: dialect.${match[1]}`);
    }
  }
  assert.ok(reads > 50, `found only ${reads} dialect reads - the pattern stopped matching`);
  assert.deepEqual(unknown, []);
});
