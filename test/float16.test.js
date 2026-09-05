import { describe, expect, it } from "./harness.js";
import { concatenateAs, float32ToFloat16Array, numberToFloat16, writeInto }
  from "../src/runtime/float16.js";

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


/**
 * 🔴 TWO IMPLEMENTATIONS, AND THE FAST ONE IS THE DEFAULT. float32ToFloat16Array
 * uses the engine's Float16Array where there is one, because the scalar loop is
 * most of a second on the 319 million floats an AF3 fold converts. That is only
 * safe if the two agree exactly, so this holds them to each other on the
 * patterns where half precision is decided - and skips itself on a runtime that
 * has only one of them, rather than passing vacuously.
 */
describe("float32ToFloat16Array against the scalar reference", () => {
  it("agrees bit for bit wherever both implementations exist", () => {
    if (typeof globalThis.Float16Array !== "function") return;
    const probe = new Float32Array([
      0, -0, 1, -1, 0.5, -0.5, 65504, 65520, 65536, 1e30, -1e30,
      Number.NaN, Infinity, -Infinity, 1 / 3, 0.1, 1023.5, 2048.5,
      6.103515625e-5, 5.960464477539063e-8, 2.9802322387695312e-8, 1e-8,
      // ...ties, which are the case a "round half up" implementation gets wrong
      1.0009765625, 1.00146484375, 1.0029296875,
    ]);
    const fast = float32ToFloat16Array(probe);
    for (let index = 0; index < probe.length; index += 1) {
      expect(fast[index]).toBe(numberToFloat16(probe[index]));
    }
  });

  it("agrees on arbitrary values too, where a hand-picked list would not look", () => {
    if (typeof globalThis.Float16Array !== "function") return;
    const values = new Float32Array(4096);
    let state = 12345;
    for (let index = 0; index < values.length; index += 1) {
      state = (state * 1103515245 + 12345) & 0x7fffffff;
      values[index] = (state / 0x7fffffff - 0.5) * 10 ** ((index % 20) - 10);
    }
    const fast = float32ToFloat16Array(values);
    let disagreements = 0;
    for (let index = 0; index < values.length; index += 1) {
      if (fast[index] !== numberToFloat16(values[index])) disagreements += 1;
    }
    expect(disagreements).toBe(0);
  });
});

describe("writeInto", () => {
  // 🔴 THE TAIL IS THE WHOLE RISK. It copies eight elements at a time and then
  // whatever is left, so a length that is not a multiple of eight is the case
  // that gets dropped - and a dropped tail is a weight buffer of the right
  // LENGTH holding zeros at the end, which no shader can notice.
  const lengths = [0, 1, 7, 8, 9, 15, 16, 17, 33];

  it("agrees with set, at every length around the unrolled step", () => {
    for (const length of lengths) {
      const source = new Float32Array(length);
      for (let index = 0; index < length; index += 1) source[index] = index * 0.3 - 2;
      for (const offset of [0, 3]) {
        const viaSet = concatenateAs("f16", length + offset + 2, (target) => {
          target.set(source, offset);
        });
        const viaWrite = concatenateAs("f16", length + offset + 2, (target) => {
          writeInto(target, source, offset);
        });
        expect(Array.from(viaWrite)).toEqual(Array.from(viaSet));
      }
    }
  });

  it("leaves an f32 target to set, which is already a memmove there", () => {
    const source = new Float32Array([1.5, -2.25, 3]);
    const target = new Float32Array(5);
    writeInto(target, source, 1);
    expect(Array.from(target)).toEqual([0, 1.5, -2.25, 3, 0]);
  });
});
