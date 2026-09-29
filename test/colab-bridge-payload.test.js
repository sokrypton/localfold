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
