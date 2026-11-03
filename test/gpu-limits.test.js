import assert from "node:assert/strict";
import { beforeEach, afterEach, describe, it } from "./harness.js";

import { forgetLearnedLimit, learnedLimitBytes, learnFromLoss } from "../web/gpu-limits.js";

const GIB = 1024 ** 3;
const HOUR = 60 * 60 * 1000;
const AN_8_GIB_CARDS_BUDGET = 4.8 * GIB;

describe("what lost devices teach this browser", () => {
  beforeEach(() => {
    const kept = new Map();
    globalThis.localStorage = {
      getItem: (key) => kept.get(key) ?? null,
      setItem: (key, value) => { kept.set(key, String(value)); },
      removeItem: (key) => { kept.delete(key); },
    };
  });
  afterEach(() => { delete globalThis.localStorage; });

  it("takes nothing from one loss, which another program can cause", () => {
    assert.equal(learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, 0), undefined);
    assert.equal(learnedLimitBytes(), undefined);
  });

  it("sets a ceiling at the second loss in an hour", () => {
    learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, 0);
    const taught = learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, HOUR - 1);
    assert.equal(taught, Math.round(2.4 * GIB));
    assert.equal(learnedLimitBytes(), taught);
  });

  it("starts again when the two are hours apart", () => {
    learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, 0);
    assert.equal(learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, HOUR), undefined);
    assert.equal(learnedLimitBytes(), undefined);
    assert.equal(typeof learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, HOUR + 1), "number");
  });

  it("does not count a loss that held little of what a fold may hold", () => {
    learnFromLoss(1 * GIB, AN_8_GIB_CARDS_BUDGET, 0);
    assert.equal(learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, 1), undefined);
    assert.equal(learnedLimitBytes(), undefined);
  });

  it("needs two more once a ceiling is set, and never raises it", () => {
    learnFromLoss(4 * GIB, AN_8_GIB_CARDS_BUDGET, 0);
    const first = learnFromLoss(4 * GIB, AN_8_GIB_CARDS_BUDGET, 1);
    assert.equal(learnFromLoss(4.5 * GIB, AN_8_GIB_CARDS_BUDGET, 2), undefined);
    assert.equal(learnFromLoss(4.5 * GIB, AN_8_GIB_CARDS_BUDGET, 3), first);
    assert.equal(learnedLimitBytes(), first);
  });

  it("gives a ceiling back when asked", () => {
    learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, 0);
    learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, 1);
    forgetLearnedLimit();
    assert.equal(learnedLimitBytes(), undefined);
  });

  it("learns nothing where nothing can be kept, and does not throw", () => {
    globalThis.localStorage = {
      getItem: () => { throw new Error("denied"); },
      setItem: () => { throw new Error("denied"); },
      removeItem: () => { throw new Error("denied"); },
    };
    assert.equal(learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, 0), undefined);
    assert.equal(learnFromLoss(3 * GIB, AN_8_GIB_CARDS_BUDGET, 1), undefined);
  });
});
