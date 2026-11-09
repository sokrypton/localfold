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

// 🔴 AND EVERY CONVENTION WebGPU READS, WebGL2 READS TOO. protenix2's MC dropout on the recycled pair term
// (`recycleMcDropout`) was implemented in webgpu/af3 and absent from webgl/af3, so on the seeds that draw it - the page's
// default among them, `Number(seed) || 1` - a WebGL2 protenix2 fold was another fold (2.2 A from WebGPU's, z_init 0.20
// off). A key may be left out only by naming the family it serves and that this backend refuses it.
const WEBGL_REFUSES = {};

test("every dialect key webgpu/af3 reads, webgl/af3 reads too", () => {
  const read = (dir) => {
    const keys = new Set();
    for (const file of walk(dir)) {
      const code = readFileSync(file, "utf8").replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
      for (const match of code.matchAll(/\bdialect\??\.(\w+)/g)) if (match[1] !== "js") keys.add(match[1]);
    }
    return keys;
  };
  const webgpu = read("webgpu/af3");
  const webgl = read("webgl/af3");
  assert.ok(webgpu.size > 20, `found only ${webgpu.size} keys webgpu/af3 reads - the pattern stopped matching`);
  assert.deepEqual([...webgpu].filter((key) => !webgl.has(key) && !(key in WEBGL_REFUSES)), []);
});
