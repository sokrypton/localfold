/**
 * The trunk cache's size bound.
 *
 * 🔴 WHAT THIS PROTECTS IS A HEAP, NOT A HIT RATE. The cache holds
 * `tokens^2 x 128` float32 so a re-sample can skip the trunk, which is worth
 * having at 58 residues and costs the page its heap at 1530 - measured, cartoon
 * mode refused after a 1530-residue fold because the tab was over Chrome's heap
 * limit. So the sizes below are the real ones, and the assertions are that the
 * small fold is kept and the large one is not.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import {
  shouldCacheTrunk, trunkCacheBytes, trunkCacheLimit,
  TRUNK_CACHE_FALLBACK_BYTES,
} from "../web/trunk-cache.js";

/** A reusable trunk for a fold of `tokens` tokens, at the real widths. */
const trunkOf = (tokens) => ({
  targetFeat: new Float32Array(tokens * 449),
  trunk: {
    pair: new Float32Array(tokens * tokens * 128),
    single: new Float32Array(tokens * 384),
  },
  recycles: 0,
});

/** Chrome's, roughly, on a 64-bit desktop. */
const CHROME = { memory: { jsHeapSizeLimit: 4 * 1024 * 1024 * 1024 } };

describe("the trunk cache's bound", () => {
  it("counts every array the trunk carries, not just the pair", () => {
    const reusable = trunkOf(58);
    const bytes = trunkCacheBytes(reusable);
    assert.equal(bytes, 58 * 449 * 4 + 58 * 58 * 128 * 4 + 58 * 384 * 4);
  });

  it("survives a trunk that is missing, or empty", () => {
    assert.equal(trunkCacheBytes(undefined), 0);
    assert.equal(trunkCacheBytes({}), 0);
  });

  it("keeps a small fold, which is what the cache is for", () => {
    const { keep, bytes } = shouldCacheTrunk(trunkOf(58), CHROME);
    assert.ok(keep);
    assert.ok(bytes < 5 * 1024 * 1024, `${bytes} bytes at 58 residues`);
  });

  // 🔴 THE FOLD THAT CAUSED THIS. 1530 residues is 1.2 GB in `pair` alone.
  it("refuses the fold that emptied the heap", () => {
    const { keep, bytes } = shouldCacheTrunk(trunkOf(1530), CHROME);
    assert.ok(!keep);
    assert.ok(bytes > 1024 * 1024 * 1024, `${bytes} bytes at 1530 residues`);
  });

  it("falls back to a fixed cap where the heap size is not reported", () => {
    assert.equal(trunkCacheLimit(undefined), TRUNK_CACHE_FALLBACK_BYTES);
    assert.equal(trunkCacheLimit({}), TRUNK_CACHE_FALLBACK_BYTES);
    assert.equal(trunkCacheLimit({ memory: {} }), TRUNK_CACHE_FALLBACK_BYTES);
    // ...and a nonsense limit is not divided by eight and believed.
    assert.equal(trunkCacheLimit({ memory: { jsHeapSizeLimit: 0 } }),
                 TRUNK_CACHE_FALLBACK_BYTES);
  });

  // The crossover, so a change to either constant has to be deliberate.
  it("holds a fold up to about a thousand residues on a 4 GB heap", () => {
    assert.ok(shouldCacheTrunk(trunkOf(1000), CHROME).keep);
    assert.ok(!shouldCacheTrunk(trunkOf(1100), CHROME).keep);
  });
});
