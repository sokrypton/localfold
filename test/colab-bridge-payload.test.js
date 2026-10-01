import { readFileSync } from "node:fs";
import { describe, expect, it } from "./harness.js";

// 🔴 A REMOTE AF2 FOLD AT 255 RESIDUES WAS ~1 GB AS JSON and died on "Invalid
// string length" with its structure drawn: every recycle carried its pair
// representation and the PAE/lDDT logits. The readback leaves them out; the
// reader never reads them. Structural, because the failure shows only at a
// length no CPU test folds.
describe("the Colab bridge's readback", () => {
  const source = readFileSync(new URL("../web/colab-bridge.js", import.meta.url), "utf8");

  it("names the model intermediates it keeps on the runtime", () => {
    const found = source.match(/const RUNTIME_ONLY = new Set\(\[([^\]]*)\]\)/);
    expect(found !== null).toBe(true);
    for (const key of ["pair", "paeLogits", "lddtLogits"]) {
      expect(found[1].includes(`"${key}"`)).toBe(true);
    }
  });

  it("drops them in the replacer that serialises the prediction", () => {
    expect(/JSON\.stringify\(pred,[^;]*RUNTIME_ONLY\.has\(key\)/s.test(source)).toBe(true);
  });
});

describe("a typed array across the bridge", () => {
  it("comes back as the same kind with the same bits, NaN included", async () => {
    globalThis.location ??= { search: "", href: "http://localhost/", origin: "http://localhost" };
    globalThis.window ??= globalThis;
    globalThis.document ??= { getElementById: () => null, addEventListener() {}, querySelector: () => null };
    const { bytesToBase64, revivePrediction } = await import("../web/colab-bridge.js");
    const sent = { pae: new Float32Array([1.5, -2.25, 3e-7, NaN]), ids: new Int32Array([1, -2, 3]) };
    const json = JSON.stringify(sent, (key, value) => ArrayBuffer.isView(value)
      ? { __typed: value.constructor.name, b64: bytesToBase64(value) } : value);
    const back = revivePrediction(json);
    expect(back.pae.constructor.name).toBe("Float32Array");
    expect(Array.from(new Uint32Array(back.pae.buffer))).toEqual(Array.from(new Uint32Array(sent.pae.buffer)));
    expect(Array.from(back.ids)).toEqual([1, -2, 3]);
    // ...and the older number-list form still reads.
    expect(Array.from(revivePrediction('{"a":{"__typed":"Uint8Array","v":[7,8]}}').a)).toEqual([7, 8]);
  });
});

describe("shared objects across the bridge", () => {
  it("are sent once and come back as the same object", async () => {
    globalThis.location ??= { search: "", href: "http://localhost/", origin: "http://localhost" };
    globalThis.window ??= globalThis;
    globalThis.document ??= { getElementById: () => null, addEventListener() {}, querySelector: () => null };
    const { encodePrediction, revivePrediction } = await import("../web/colab-bridge.js");
    // An AF2 prediction's shape: each pass wraps its own result, and
    // contactSource is one of those results again.
    const structure = { atom37: new Float32Array(3000).fill(1.25) };
    const confidence = { plddt: new Float32Array([90, 80]), predictedAlignedError: new Float32Array(400) };
    const pass = { structure, confidence, contactProbs: new Float32Array(400).fill(0.5), pair: new Float32Array(9) };
    const pred = { recycles: [{ structure, confidence, pass }], contactSource: pass, scores: { plddt: [90, 80] } };
    const json = encodePrediction(pred);
    const back = revivePrediction(json);
    expect(back.contactSource).toBe(back.recycles[0].pass);
    expect(back.recycles[0].structure).toBe(back.recycles[0].pass.structure);
    expect(back.recycles[0].pass.confidence.plddt).toBe(back.recycles[0].confidence.plddt);
    expect(back.recycles[0].structure.atom37.constructor.name).toBe("Float32Array");
    expect(back.recycles[0].structure.atom37[2999]).toBe(1.25);
    expect(back.contactSource.pair).toBe(undefined);        // RUNTIME_ONLY still stays home
    expect("__id" in back.recycles[0]).toBe(false);
    // ...and the repeats cost a reference, not a copy: against the same
    // prediction with every shared object written out each time, which is
    // what a plain stringify did.
    const { bytesToBase64 } = await import("../web/colab-bridge.js");
    const copies = JSON.stringify(pred, (key, value) => key === "pair" ? undefined
      : ArrayBuffer.isView(value) ? { __typed: value.constructor.name, b64: bytesToBase64(value) } : value);
    expect(json.length * 2 < copies.length).toBe(true);
    expect((json.match(/"__ref"/g) ?? []).length).toBe(3);   // pass.structure, pass.confidence, contactSource
  });
});
