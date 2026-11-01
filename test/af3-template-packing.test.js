/**
 * Per-chain template slots, packed as AlphaFold 3 packs them (model/features.py,
 * `Templates.compute_features`): slot k holds every chain's k-th template at that chain's own
 * tokens, each chain keeps its first four, and a slot spanning chains keeps its own.
 *
 * Found by a homomer: six copies with one template each were six slots, and the trunk's four
 * refused them ("6 templates for 4 slots"). In AlphaFold 3 they are one.
 */
import assert from "node:assert/strict";
import { describe, it } from "./harness.js";
import { packTemplateSlots } from "../web/af3-model.js";

const DENSE = 24;
// a slot covering `tokens` (a list of token indices) out of `total`, its aatype set to `tag`
const slot = (total, tokens, tag, extra = {}) => {
  const aatype = new Int32Array(total).fill(31);
  const atomMask = new Float32Array(total * DENSE);
  const atomPositions = new Float32Array(total * DENSE * 3);
  for (const token of tokens) {
    aatype[token] = tag;
    atomMask[token * DENSE] = 1;
    atomPositions[token * DENSE * 3] = tag;
  }
  return { aatype, atomMask, atomPositions, covered: tokens.length, atoms: tokens.length, spanChains: false, ...extra };
};

describe("AF3 template slots", () => {
  it("are one slot for six copies with one template each", () => {
    const built = Array.from({ length: 6 }, (_, chain) => ({ chain, slot: slot(12, [chain * 2, chain * 2 + 1], chain) }));
    const packed = packTemplateSlots(built);
    assert.equal(packed.length, 1);
    assert.deepEqual(Array.from(packed[0].aatype), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    assert.equal(packed[0].covered, 12);
    // the chains do not see each other: the merged slot is not one that spans them
    assert.equal(packed[0].spanChains, false);
  });

  it("put a chain's second template in the second slot", () => {
    const built = [
      { chain: 0, slot: slot(4, [0, 1], 1) },
      { chain: 0, slot: slot(4, [0, 1], 2) },
      { chain: 1, slot: slot(4, [2, 3], 3) },
    ];
    const packed = packTemplateSlots(built);
    assert.equal(packed.length, 2);
    assert.deepEqual(Array.from(packed[0].aatype), [1, 1, 3, 3]);
    assert.deepEqual(Array.from(packed[1].aatype), [2, 2, 31, 31]);
  });

  it("keep a chain's first four, as AF3 does", () => {
    const built = Array.from({ length: 6 }, (_, k) => ({ chain: 0, slot: slot(2, [0, 1], k) }));
    const packed = packTemplateSlots(built);
    assert.equal(packed.length, 4);
    assert.deepEqual(packed.map((s) => s.aatype[0]), [0, 1, 2, 3]);
  });

  it("leave a slot spanning chains on its own", () => {
    const built = [
      { chain: 0, slot: slot(4, [0, 1], 1) },
      { chain: 0, slot: slot(4, [0, 1, 2, 3], 9, { spanChains: true }) },
    ];
    const packed = packTemplateSlots(built);
    assert.equal(packed.length, 2);
    assert.equal(packed[1].spanChains, true);
    assert.deepEqual(Array.from(packed[1].aatype), [9, 9, 9, 9]);
  });
});
