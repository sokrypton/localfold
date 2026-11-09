import { test } from "node:test";
import assert from "node:assert/strict";
import { fitsFloat16, float32ToFloat16Array, float32ToFloat16IfFits, float32ToFloat16Scalar } from "../shared/weights/float16.js";

// `float32ToFloat16IfFits` must be `fitsFloat16` then `float32ToFloat16Array`, padded with zeros - on whichever path
// this runtime takes - and refuse exactly what `fitsFloat16` refuses.
test("float32ToFloat16IfFits is the check, then the conversion, then zeros", () => {
  let seed = 3;
  const random = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const values = Float32Array.from({ length: 4099 }, () => (random() - 0.5) * 2 ** (random() * 40 - 25));
  values[7] = 65504; values[8] = -65504; values[9] = 6.1e-6; values[10] = -0;
  const padded = float32ToFloat16IfFits(values, 4104);
  const plain = float32ToFloat16Scalar(values);
  assert.deepEqual(Array.from(float32ToFloat16Array(values)), Array.from(plain));
  for (let i = 0; i < values.length; i += 1) assert.equal(padded[i], plain[i], `element ${i}`);
  for (let i = values.length; i < 4104; i += 1) assert.equal(padded[i], 0);
  for (const bad of [65505, 70000, Infinity, NaN, -1e9]) {
    const withBad = Float32Array.from(values); withBad[4000] = bad;
    assert.equal(fitsFloat16(withBad), false);
    assert.equal(float32ToFloat16IfFits(withBad), null, `refuses ${bad}`);
  }
});
