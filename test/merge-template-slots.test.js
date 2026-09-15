import test from "node:test";
import assert from "node:assert/strict";
import { mergeTemplateSlots, GAP_AATYPE } from "../src/af3/featurise/template-input.js";
import { NUM_DENSE, multichainMaskFor } from "../src/af3/featurise/template-features.js";

/** A slot covering one token range, one atom per covered token. */
function slotOver(tokens, from, count, aatype) {
  const slot = {
    aatype: new Int32Array(tokens).fill(GAP_AATYPE),
    atomPositions: new Float32Array(tokens * NUM_DENSE * 3),
    atomMask: new Float32Array(tokens * NUM_DENSE),
    covered: count,
    atoms: count,
  };
  for (let token = from; token < from + count; token += 1) {
    slot.aatype[token] = aatype;
    slot.atomMask[token * NUM_DENSE] = 1;
    slot.atomPositions[token * NUM_DENSE * 3] = token;
  }
  return slot;
}

test("mergeTemplateSlots", async (t) => {
  await t.test("carries every chain's coverage into one slot", () => {
    const merged = mergeTemplateSlots([slotOver(10, 0, 4, 3), slotOver(10, 4, 6, 7)]);
    assert.equal(merged.covered, 10);
    assert.equal(merged.atoms, 10);
    for (let token = 0; token < 10; token += 1) {
      assert.equal(merged.aatype[token], token < 4 ? 3 : 7, `aatype at ${token}`);
      assert.equal(merged.atomMask[token * NUM_DENSE], 1);
      assert.equal(merged.atomPositions[token * NUM_DENSE * 3], token);
    }
  });

  // 🔴 THE FALSIFICATION: two slots claiming one token must raise rather than
  // letting the later one win, because a silent overwrite is a template that
  // says one chain is where the other is.
  await t.test("refuses slots that overlap", () => {
    assert.throws(() => mergeTemplateSlots([slotOver(10, 0, 5, 3), slotOver(10, 4, 6, 7)]),
                  /both cover token 4/);
  });

  await t.test("refuses slots of different lengths", () => {
    assert.throws(() => mergeTemplateSlots([slotOver(10, 0, 4, 3), slotOver(12, 4, 6, 7)]),
                  /disagree on token count/);
  });

  await t.test("a lone slot is returned unchanged", () => {
    const one = slotOver(10, 0, 4, 3);
    assert.equal(mergeTemplateSlots([one]), one);
  });

  // 🔴 WHY IT EXISTS AT ALL. `multichainMaskFor` opens a cross-chain pair only
  // where the slot covers BOTH ends, so two per-chain slots carry no interface
  // however `spanChains` is set - which is the whole reason a complex wants one
  // merged slot.
  await t.test("only a merged slot can open a cross-chain pair", () => {
    const asymId = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    const coverageOf = (slot) => Array.from({ length: 10 }, (_, token) =>
      (slot.atomMask[token * NUM_DENSE] > 0 ? 1 : 0));
    const chainA = slotOver(10, 0, 4, 3);
    const merged = mergeTemplateSlots([chainA, slotOver(10, 4, 6, 7)]);
    const across = (slot) => multichainMaskFor(asymId, 10,
      { coverage: coverageOf(slot), spanChains: true })[0 * 10 + 5];
    assert.equal(across(chainA), 0);
    assert.equal(across(merged), 1);
  });
});
