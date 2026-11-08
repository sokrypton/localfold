// The scalar float32 -> float16 conversion (what a browser without Float16Array - Safari before 18.2 - runs for every
// weight) against the platform's own, bit for bit; and `fitsFloat16` against the rule it replaced.
import { test } from "node:test";
import assert from "node:assert/strict";
import { fitsFloat16, float32ToFloat16Scalar, numberToFloat16 } from "../shared/weights/float16.js";

const edges = [0, -0, 1, -2, 0.5, 65504, -65504, 65519, 65520, 65536, 1e-8, 6e-8, 5.96e-8, 6.1e-5, 6.104e-5,
  Infinity, -Infinity, NaN, 2 ** -24, 2 ** -25, 2 ** -14, 1 + 2 ** -11, 1 + 3 * 2 ** -11, 3.14159, -7.5e-5];

test("the scalar conversion is the per-number one, bit for bit", () => {
  const values = new Float32Array(200000);
  let seed = 7;
  for (let i = 0; i < values.length; i += 1) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    values[i] = (seed / 2147483648 - 0.5) * 2 ** ((i % 48) - 24);
  }
  values.set(edges, 0);
  const fast = float32ToFloat16Scalar(values);
  for (let i = 0; i < values.length; i += 1) {
    if (fast[i] !== numberToFloat16(values[i])) assert.fail(`${values[i]}: ${fast[i]} against ${numberToFloat16(values[i])}`);
  }
});

test("the scalar conversion is the native one where there is one", { skip: typeof Float16Array !== "function" }, () => {
  const values = Float32Array.from({ length: 100000 }, (_, i) => Math.sin(i * 1.3) * 2 ** ((i % 40) - 20));
  values.set(edges, 0);
  const native = new Uint16Array(new Float16Array(values).buffer);
  const fast = float32ToFloat16Scalar(values);
  for (let i = 0; i < values.length; i += 1) {
    if (Number.isNaN(values[i])) { assert.equal(fast[i] & 0x7c00, 0x7c00); continue; }
    if (fast[i] !== native[i]) assert.fail(`${values[i]}: ${fast[i]} against native ${native[i]}`);
  }
});

test("fitsFloat16 is |x| <= 65504 with NaN refused", () => {
  const rule = (array) => { let largest = 0; for (const v of array) largest = Math.max(largest, Math.abs(v)); return largest <= 65504 && !array.some(Number.isNaN); };
  for (const array of [[0, 1, -65504], [65504], [65504.5], [-70000], [NaN], [Infinity], [1e-30, -3]]) {
    assert.equal(fitsFloat16(new Float32Array(array)), rule(new Float32Array(array)), JSON.stringify(array));
  }
});
