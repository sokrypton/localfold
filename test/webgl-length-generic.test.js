import test from "node:test";
import assert from "node:assert/strict";
import { genericSource, learnLengthNames, lengthSkeleton } from "../webgl/runtime/length-generic.js";

const kernel = (rows, channels, scale = "") => `#version 300 es
const int W = 3072;
const int CHANNELS = ${channels};
const int ROWS = ${rows};
const int ELEMENTS = ${rows * channels};
const int HALF = ROWS / 2;
${scale}
void main() {
  int i = ELEMENTS + HALF + CHANNELS + W;
}`;

test("the constants that follow the length are learned, widths are not", () => {
  // one kernel at two widths in each fold: the widths' set is the same at both lengths
  const known = learnLengthNames([kernel(60, 128), kernel(60, 64)], [kernel(127, 128), kernel(127, 64)], new Map());
  assert.equal(known.size, 1);
  const [names] = known.values();
  assert.deepEqual([...names].sort(), ["ELEMENTS", "ROWS"]);
});

test("a kernel at only one of the lengths teaches nothing", () => {
  assert.equal(learnLengthNames([kernel(60, 128)], [kernel(127, 128, "// another kernel")], new Map()).size, 0);
});

test("moving constants become uniforms, constants computed from them #defines", () => {
  const generic = genericSource(kernel(100, 128), new Set(["ROWS", "ELEMENTS"]));
  assert.deepEqual(generic.values, [["ROWS", 100], ["ELEMENTS", 12800]]);
  assert.match(generic.source, /^uniform int ROWS;$/m);
  assert.match(generic.source, /^uniform int ELEMENTS;$/m);
  assert.match(generic.source, /^#define HALF \(ROWS \/ 2\)$/m);
  assert.match(generic.source, /^const int CHANNELS = 128;$/m);
  // every length's generic source is the same program
  assert.equal(genericSource(kernel(7, 128), new Set(["ROWS", "ELEMENTS"])).source, generic.source);
  assert.equal(lengthSkeleton(kernel(7, 128)).skeleton, lengthSkeleton(kernel(100, 128)).skeleton);
});

test("a float computed from a moving constant is refused (not bit-exact at run time)", () => {
  const source = kernel(100, 128, "const float INV = 1.0 / float(ROWS);");
  assert.equal(genericSource(source, new Set(["ROWS"])), null);
});

test("nothing moving, nothing generic", () => {
  assert.equal(genericSource(kernel(100, 128), new Set(["NOPE"])), null);
});
