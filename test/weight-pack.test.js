/**
 * Every weight pack writes what it reserves.
 *
 * 🔴 THIS IS THE GATE FOR THE BUG THAT COST AN HOUR AND ALMOST BECAME A
 * PUBLISHED CONCLUSION. `packOuterProductMeanWeights` reserved its offsets over
 * `ORDER + OPTIONAL` and WROTE over `ORDER` alone, so rosettafold3's projection
 * bias was in the generated WGSL, in the offsets, and NOT in the buffer. The
 * shader read a region of zeros, added zero, and the fold came out
 * bit-identical to one with no bias - so every oracle seam matched the previous
 * run to the last digit, which reads exactly like "this convention does not
 * matter for this model".
 *
 * Nothing could see it. A fold could not (bit-identical), an oracle seam could
 * not (bit-identical), and reading the function twice did not. What sees it is
 * this: pack a tensor of DISTINCT, NON-ZERO values and read it back at the
 * offset the pack reported. A reserved-but-unwritten region reads as zeros.
 *
 * It is a property test rather than a digest, so a legitimate layout change
 * does not have to be re-recorded, and a failure names the tensor.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { packOuterProductMeanWeights } from "../src/af3/outer-product-mean-webgpu.js";
import { packGridAttentionWeights } from "../src/af3/grid-attention-webgpu.js";
import { packTransitionWeights } from "../src/af3/transition-webgpu.js";
import { packNamedWeights } from "../src/runtime/weight-pack.js";

/**
 * A tensor whose every element is distinct and non-zero, tagged by `mark`, so a
 * value read back at the wrong offset - or out of an unwritten region - cannot
 * coincide with the right one.
 */
let nextMark = 1;
function tensor(length) {
  const mark = nextMark;
  nextMark += 1;
  return Float32Array.from({ length }, (_, index) => mark * 1000 + index + 1);
}

/**
 * Every packed tensor is readable at its own offset.
 *
 * 🔴 `packing` IS THE LIST THE PACK ACTUALLY USED, not the caller's idea of it.
 * Checking against the caller's list would pass a pack that quietly dropped a
 * name from both loops together - which is a different bug and also real.
 */
function assertRoundTrip(result, weights, label) {
  const names = result.packing ?? Object.keys(result.offsets);
  assert.ok(names.length > 0, `${label}: packed nothing`);
  for (const name of names) {
    const offset = result.offsets[name];
    assert.equal(typeof offset, "number", `${label}: ${name} has no offset`);
    const source = weights[name];
    if (source === undefined || source === null) continue;
    for (let index = 0; index < source.length; index += 1) {
      assert.equal(result.data[offset + index], source[index],
        `${label}: ${name} element ${index} is not at its offset`
        + ` (${result.data[offset + index]} against ${source[index]})`
        + (result.data[offset + index] === 0
          ? " - a ZERO here means the offset was reserved and never written" : ""));
    }
  }
}

describe("weight packs write what they reserve", () => {
  it("the outer product mean, with rosettafold3's optional biases", () => {
    const weights = {
      layerNormInputScale: tensor(8), layerNormInputOffset: tensor(8),
      leftProjection: tensor(8 * 4), rightProjection: tensor(8 * 4),
      outputW: tensor(16 * 6), outputB: tensor(6),
      // The two that were reserved and never written.
      leftProjectionBias: tensor(4), rightProjectionBias: tensor(4),
    };
    const result = packOuterProductMeanWeights(weights);
    assertRoundTrip(result, weights, "outer product mean");
    assert.ok(result.offsets.leftProjectionBias !== undefined,
      "the optional bias was not packed at all");
  });

  it("...and without them, which must pack strictly less", () => {
    const base = {
      layerNormInputScale: tensor(8), layerNormInputOffset: tensor(8),
      leftProjection: tensor(8 * 4), rightProjection: tensor(8 * 4),
      outputW: tensor(16 * 6), outputB: tensor(6),
    };
    const bare = packOuterProductMeanWeights(base);
    assertRoundTrip(bare, base, "outer product mean, bias-free");
    assert.equal(bare.offsets.leftProjectionBias, undefined);
    // 🔴 AND `null` IS HOW THE LOADER SAYS ABSENT, not `undefined`. A strict
    // `=== undefined` here is what killed boltz2 in a path it does not share.
    const withNulls = packOuterProductMeanWeights(
      { ...base, leftProjectionBias: null, rightProjectionBias: null });
    assert.equal(withNulls.data.length, bare.data.length,
      "a null optional tensor must pack as absent, not as an empty slot");
  });

  it("grid attention, whose qkvg slot is four tensors interleaved", () => {
    const heads = 2;
    const dimension = 4;
    const width = heads * dimension;
    const channels = 6;
    const weights = {
      heads, dimension,
      actNormScale: tensor(channels), actNormOffset: tensor(channels),
      pairBiasProjection: tensor(channels * heads),
      qProjection: tensor(channels * width), kProjection: tensor(channels * width),
      vProjection: tensor(channels * width), gatingQuery: tensor(channels * width),
      outputProjection: tensor(width * channels),
      gatingQueryBias: tensor(width), outputProjectionBias: tensor(channels),
    };
    const result = packGridAttentionWeights(weights);
    // The qkvg slot is interleaved and transposed, so its four sources are not
    // readable elementwise - every OTHER name must be, and the two optional
    // biases are the ones that matter here.
    for (const name of ["gatingQueryBias", "outputProjectionBias"]) {
      const offset = result.offsets[name];
      assert.equal(typeof offset, "number", `${name} was not packed`);
      for (let index = 0; index < weights[name].length; index += 1) {
        assert.equal(result.data[offset + index], weights[name][index],
          `grid attention: ${name} element ${index} is not at its offset`);
      }
    }
  });

  it("the transition", () => {
    const weights = {
      layerNormScale: tensor(8), transitionW: tensor(8 * 16), outputW: tensor(8 * 8),
    };
    let result;
    try { result = packTransitionWeights(weights); }
    catch (error) {
      // The order is the module's; if it wants more names, say which rather
      // than passing by accident.
      assert.match(error.message, /missing/, error.message);
      return;
    }
    assertRoundTrip(result, weights, "transition");
  });
});

describe("packNamedWeights", () => {
  it("reserves and writes over ONE list, so the offsets cover the data exactly", () => {
    const weights = { a: tensor(3), b: tensor(5), maybe: tensor(7) };
    const result = packNamedWeights(weights,
      { label: "test", order: ["a", "b"], optional: ["maybe"] });
    assert.deepEqual(result.packing, ["a", "b", "maybe"]);
    assert.equal(result.data.length, 3 + 5 + 7);
    assertRoundTrip(result, weights, "packNamedWeights");
  });

  it("treats null as absent and undefined as an error", () => {
    const weights = { a: tensor(3), b: tensor(5), maybe: null };
    const result = packNamedWeights(weights,
      { label: "test", order: ["a", "b"], optional: ["maybe"] });
    assert.equal(result.data.length, 8);
    assert.equal(result.offsets.maybe, undefined);
    assert.throws(() => packNamedWeights({ a: tensor(3) },
      { label: "test", order: ["a", "b"] }), /test missing b/);
  });

  it("cannot be given two lists, which is the whole point", () => {
    // There is no second list to pass: the signature takes `order` and
    // `optional` and derives `packing` itself. This asserts the derived list is
    // what both loops used, by checking the LAST tensor is readable - the one a
    // reserve-only bug leaves as zeros.
    const weights = { a: tensor(3), tail: tensor(4) };
    const result = packNamedWeights(weights,
      { label: "test", order: ["a"], optional: ["tail"] });
    assert.equal(result.data[result.offsets.tail], weights.tail[0]);
    assert.notEqual(result.data[result.offsets.tail], 0);
  });
});

describe("carriesTensor, the presence rule that cost 7x", () => {
  it("asks the SOURCES map and never touches the value", async () => {
    const { SOURCES, carriesTensor } = await import("../src/runtime/weight-sources.js");
    let decodes = 0;
    const weights = {
      [SOURCES]: { present: { count: 4 }, absent: null },
      // A bound field is a GETTER that decodes when read. If the predicate
      // touches it, this counter moves - which is the 38.5 s fold.
      get present() { decodes += 1; return new Float32Array(4); },
      get absent() { decodes += 1; return null; },
    };
    assert.equal(carriesTensor(weights, "present"), true);
    assert.equal(carriesTensor(weights, "absent"), false);
    assert.equal(decodes, 0, "the predicate DECODED a tensor to ask whether it exists");
  });

  it("falls back to the value when there is no SOURCES map", async () => {
    const { carriesTensor } = await import("../src/runtime/weight-sources.js");
    assert.equal(carriesTensor({ a: new Float32Array(2) }, "a"), true);
    assert.equal(carriesTensor({ a: undefined }, "a"), false);
    // 🔴 `null` IS HOW THE LOADER SAYS ABSENT. A strict `!== undefined` here is
    // what killed boltz2 in a path it does not share.
    assert.equal(carriesTensor({ a: null }, "a"), false);
    assert.equal(carriesTensor(undefined, "a"), false);
  });

  it("is what the five former predicates now call", async () => {
    const { blockHasUpGate, blockHasKqNorm } =
      await import("../src/af3/atom-encoder-webgpu.js");
    const { txHasUpGate, txHasKqNorm } =
      await import("../src/af3/diffusion-transformer-webgpu.js");
    const { hasBondTypes } = await import("../src/af3/embedder-webgpu.js");
    const { SOURCES } = await import("../src/runtime/weight-sources.js");
    const block = { [SOURCES]: { ffwAToB: { count: 1 }, queryLayerNormScale: null } };
    assert.equal(blockHasUpGate(block), true);
    assert.equal(txHasUpGate(block), true);
    assert.equal(blockHasKqNorm(block), false);
    assert.equal(txHasKqNorm(block), false);
    assert.equal(hasBondTypes({ [SOURCES]: { tokenBondsTypeEmbed: { count: 1 } } }), true);
    assert.equal(hasBondTypes({ [SOURCES]: {} }), false);
  });
});
