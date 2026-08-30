import { describe, expect, it } from "./harness.js";
import { float32ToFloat16Array, numberToFloat16 } from "../src/runtime/float16.js";

describe("float16 encoding", () => {
  it("encodes IEEE-754 boundary values", () => {
    expect(numberToFloat16(0)).toBe(0x0000);
    expect(numberToFloat16(-0)).toBe(0x8000);
    expect(numberToFloat16(1)).toBe(0x3c00);
    expect(numberToFloat16(-2)).toBe(0xc000);
    expect(numberToFloat16(Infinity)).toBe(0x7c00);
    expect(numberToFloat16(-Infinity)).toBe(0xfc00);
    expect(numberToFloat16(Number.NaN) & 0x7c00).toBe(0x7c00);
  });

  it("converts arrays without changing their element count", () => {
    expect(float32ToFloat16Array(new Float32Array([1, 0.5, -4]))).toEqual(
      new Uint16Array([0x3c00, 0x3800, 0xc400]),
    );
  });

  it("uses round-to-nearest-even for float32 inputs", () => {
    const native = new Float16Array(1);
    const nativeBits = new Uint16Array(native.buffer);
    const values = new Float32Array([
      -0.748291015625,
      16.1171875,
      52.859375,
      -78.96875,
      0.000061005353927612305,
    ]);
    for (const value of values) {
      native[0] = value;
      expect(numberToFloat16(value)).toBe(nativeBits[0]);
    }
  });
});
