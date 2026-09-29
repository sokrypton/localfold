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
