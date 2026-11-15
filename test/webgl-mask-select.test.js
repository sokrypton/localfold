/**
 * A WebGL2 shader never ADDS a large mask constant to a live score.
 *
 * 🔴 THIS IS ISSUE #10 ON LINUX: "very strange protein structures", no error. The attention kernels masked a key
 * as every reference does, `logit = score + 1e9 * (mask - 1.0)`, which adds exactly zero where the mask is one.
 * NVIDIA's OpenGL compiler reassociates it into `(score + 1e9 * mask) - 1e9`, and a float near 1e9 moves in
 * steps of 64, so every logit of a LIVE key was rounded to a multiple of 64 - a softmax over garbage, with a
 * confidence number beside it. ANGLE over Vulkan and over Metal compute it as written, which is every machine
 * this backend had been checked on; a Linux reader with no WebGPU (Firefox, and Chrome on most drivers) gets
 * the desktop GL compiler. GLSL ES has no `precise` to forbid it.
 *
 * So the rule is structural: the constant is SELECTED, never summed into a live term -
 * `score + (mask == 1.0 ? 0.0 : 1e9 * (mask - 1.0))` - which is the same value for every mask and leaves the
 * compiler one addition it cannot rearrange. This fails when a large constant multiplies a parenthesis on a
 * line with no such select (webgl/README.md, "Linux").
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { sourceFiles } from "./helpers/source-files.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
// a constant of 1e4 or more multiplying a parenthesised term: the shape of an additive mask
const LARGE_TIMES_TERM = /\b\d(?:\.\d+)?e(?:[4-9]|\d\d)\s*\*\s*\(/;
const SELECTED = /==\s*1\.0\s*(?:&&|\|\|)?[^?]*\?\s*0\.0\s*:|if\s*\(\s*\w+\s*!=\s*1\.0\s*\)/;

describe("WebGL2 mask constants", () => {
  it("are selected, never added to a live score", () => {
    const offenders = [];
    let guarded = 0;
    for (const file of sourceFiles(join(ROOT, "webgl"))) {
      const lines = readFileSync(file, "utf8").split("\n");
      lines.forEach((line, at) => {
        const code = line.replace(/^\s*(\/\/|\*|\/\*).*$/, "");
        if (!LARGE_TIMES_TERM.test(code)) return;
        if (SELECTED.test(code)) { guarded += 1; return; }
        offenders.push(`${relative(ROOT, file)}:${at + 1}: ${line.trim()}`);
      });
    }
    assert.deepEqual(offenders, []);
    // a rule that stops matching passes by finding nothing
    assert.ok(guarded >= 9, `only ${guarded} guarded mask terms found - the pattern has stopped matching`);
  });
});
