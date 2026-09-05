import { describe, expect, it } from "./harness.js";
import {
  packHalfWords,
  storageArray,
  storageBytes,
  storageWords,
  storedElement,
  storedPair,
  unpackHalfWords,
} from "../src/runtime/storage.js";

describe("activation storage", () => {
  it("names the WGSL array element", () => {
    expect(storageArray("f32")).toBe("f32");
    expect(storageArray("f16")).toBe("u32");
  });

  it("refuses a storage it does not know", () => {
    expect(() => storageArray("int8")).toThrow();
    expect(() => storedElement("bf16", "a", "i")).toThrow();
  });

  it("halves the words an f16 tensor takes, rounding an odd count up", () => {
    expect(storageWords(8, "f32")).toBe(8);
    expect(storageWords(8, "f16")).toBe(4);
    expect(storageWords(9, "f16")).toBe(5);
    expect(storageBytes(9, "f16")).toBe(20);
  });

  it("parenthesises an index expression so a sum survives the shift", () => {
    expect(storedElement("f32", "src", "base + c")).toBe("src[base + c]");
    expect(storedElement("f16", "src", "base + c"))
      .toBe("unpack2x16float(src[(base + c) >> 1u])[(base + c) & 1u]");
  });

  it("writes a pair as one word packed, and as two stores plain", () => {
    expect(storedPair("f16", "out", "w", "a", "b"))
      .toBe("out[w] = pack2x16float(vec2<f32>(a, b));");
    expect(storedPair("f32", "out", "w", "a", "b"))
      .toBe("out[(w) * 2u] = a;\nout[(w) * 2u + 1u] = b;");
  });
});

describe("half-word packing", () => {
  it("puts element 2i in the low half of word i", () => {
    // 1.0 is 0x3c00 and 2.0 is 0x4000, so the first word is 0x40003c00.
    expect(packHalfWords(new Float32Array([1, 2]))).toEqual(new Uint32Array([0x40003c00]));
  });

  it("leaves the unused half of a trailing odd word at zero", () => {
    expect(packHalfWords(new Float32Array([1]))).toEqual(new Uint32Array([0x00003c00]));
  });

  it("round-trips every value a half can hold exactly", () => {
    const values = new Float32Array([
      0, -0, 1, -1, 0.5, -2, 65504, -65504, 2 ** -14, 2 ** -24, -(2 ** -24),
      Infinity, -Infinity,
    ]);
    const back = unpackHalfWords(packHalfWords(values), values.length);
    for (let index = 0; index < values.length; index += 1) {
      expect(back[index]).toBe(values[index]);
    }
    // -0 has to survive as -0, not as 0: it is the one value === cannot tell.
    expect(Object.is(back[1], -0)).toBe(true);
  });

  it("carries NaN through as a NaN", () => {
    const back = unpackHalfWords(packHalfWords(new Float32Array([Number.NaN])), 1);
    expect(Number.isNaN(back[0])).toBe(true);
  });

  it("rounds to nearest even, and to about three significant digits", () => {
    const values = new Float32Array([1 / 3, Math.PI, -1234.5678, 6e4]);
    const back = unpackHalfWords(packHalfWords(values), values.length);
    for (let index = 0; index < values.length; index += 1) {
      const error = Math.abs(back[index] - values[index]) / Math.abs(values[index]);
      expect(error < 5e-4).toBe(true);
    }
  });

  it("flushes a magnitude below the smallest subnormal to zero", () => {
    // 🔴 THE FLOOR IS 6e-8, WHICH IS NOT SMALL FOR AN ACTIVATION. A tensor
    // whose values live below it comes back as zeros and the relative error is
    // 1, not 5e-4 - so a kernel converted to this storage has to be one whose
    // activations are normalised, which every one converted so far is.
    const back = unpackHalfWords(packHalfWords(new Float32Array([1e-8, -1e-8])), 2);
    expect(back[0]).toBe(0);
    expect(Object.is(back[1], -0)).toBe(true);
  });

  it("saturates past the largest half rather than wrapping", () => {
    const back = unpackHalfWords(packHalfWords(new Float32Array([70000, -70000])), 2);
    expect(back[0]).toBe(Infinity);
    expect(back[1]).toBe(-Infinity);
  });

  it("refuses to read more halves than the words hold", () => {
    expect(() => unpackHalfWords(new Uint32Array(2), 5)).toThrow();
  });
});
