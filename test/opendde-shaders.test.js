/**
 * OpenDDE's GPU branches, asserted on the WGSL they generate.
 *
 * 🔴 A BRANCH THAT NEVER REACHES THE KERNEL AGREES WITH ITSELF, and comparing
 * cache KEYS is not comparing kernels - this repository records a unit test
 * that passed for exactly that reason while the fold came back NaN. So these
 * assert on the generated source: two arms, two different shaders, and the
 * specific line that differs.
 *
 * It matters most for the branches that turn out to be INERT on the shipped
 * path. `keyMaskedAtomAttention` changes nothing on any batch this featuriser
 * produces, because the key window is `min(128, atomCount)` and every key is
 * therefore real - so "the two arms agree to every digit" is the right answer
 * AND the answer a flag that never arrived would give. The only way to tell
 * them apart is to look at the kernel.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { createAtomBlockShaders, createAtomEncoderShaders } from "../src/af3/atom-encoder-webgpu.js";
import { createGridAttentionShaders, packGridAttentionWeights } from "../src/af3/grid-attention-webgpu.js";
import { ALPHAFOLD3, OPENDDE } from "../src/af3/dialect.js";

const ATOM_SHAPE = {
  subsets: 1, queries: 32, keys: 128, channels: 128, heads: 4, dimension: 32,
};

describe("the atom attention's mask bias", () => {
  const lineOf = (keyMaskedAtomAttention) => createAtomBlockShaders(
    "", { ...ATOM_SHAPE, keyMaskedAtomAttention },
  ).attendFor(0).split("\n").find((line) => line.includes("let bias ="));

  it("is a PRODUCT under AlphaFold 3 - both padded, or no penalty", () => {
    const line = lineOf(false);
    assert.match(line, /\(queries_mask\[query_index\] - 1\.0\) \* \(keys_mask\[key_index\] - 1\.0\)/);
  });

  it("is a SUM under OpenDDE - either padded is enough", () => {
    const line = lineOf(true);
    assert.match(line, /-1\.0e9 \* \(\(1\.0 - queries_mask\[query_index\]\) \+ /);
  });

  it("really does generate two different kernels", () => {
    // 🔴 THE ASSERTION THE NULL RESULT RESTS ON. Measured, the two arms move a
    // 68-residue trunk by exactly nothing - because no batch this featuriser
    // produces has a padded key - and that is indistinguishable from a flag
    // that was dropped on the way to the shader.
    assert.notEqual(lineOf(false), lineOf(true));
  });
});

describe("the atom-pair LayerNorm, shared or per block", () => {
  const shaderFor = (perBlockPair) => createAtomEncoderShaders(
    { ...ATOM_SHAPE, tokens: 8, dense: 8, pairChannels: 16, perTokenChannels: 384,
      trunkSingleChannels: 384, trunkPairChannels: 128, blocks: 3, perBlockPair },
    Object.fromEntries(["singleToPairCondRow", "singleToPairCondCol", "embedPairOffsets",
      "embedPairDistances", "embedPairOffsetsValid", "pairMlp1", "pairMlp2", "pairMlp3",
      "pairInputLayerNormScale", "pairLogitsProjection", "lnormTrunkSingleCondScale",
      "embedTrunkSingleCond", "lnormTrunkPairCondScale", "embedTrunkPairCond",
      "atomPositionsToFeatures", "projectAtomFeaturesForAggr"].map((n) => [n, 0])),
    {}).pairLogits;

  it("indexes the scale by block only under OpenDDE", () => {
    assert.match(shaderFor(true), /P_pairInputLayerNormScale \+ block \* C_PAIR \+ c/);
    assert.doesNotMatch(shaderFor(false), /P_pairInputLayerNormScale \+ block \* C_PAIR/);
  });

  it("generates two different kernels", () => {
    assert.notEqual(shaderFor(true), shaderFor(false));
  });
});

describe("the grid attention's bias kernel", () => {
  /**
   * 🔴 IT USED TO GENERATE AN EMPTY BODY BELOW FOUR HEADS. The heads are
   * vectorised four to a `vec4`, and `Array.from({ length: heads / 4 })` is
   * EMPTY at two - so `main` referenced none of its bindings, WebGPU inferred a
   * layout with one entry, and the bind group of three failed validation with a
   * message naming neither the kernel nor the head count. OpenDDE's template
   * stack has two heads; every head count in this repository before it was 4,
   * 8, 12 or 16.
   */
  const biasFor = (channels, heads, dimension, dialect) => {
    const weights = {
      heads, dimension,
      actNormScale: new Float32Array(channels), actNormOffset: new Float32Array(channels),
      pairBiasProjection: new Float32Array(channels * heads),
      qProjection: new Float32Array(heads * dimension * channels),
      kProjection: new Float32Array(channels * heads * dimension),
      vProjection: new Float32Array(channels * heads * dimension),
      gatingQuery: new Float32Array(channels * heads * dimension),
      outputProjection: new Float32Array(heads * dimension * channels),
    };
    return createGridAttentionShaders(
      { n: 16, channels, heads, dimension, transpose: false, residual: true,
        stagedPrecision: "f32" },
      packGridAttentionWeights(weights).offsets, 1e-5, "fast", dialect, "f32", "f32",
      { q: "f32", k: "f32", v: "f32", gate: "f32" }).bias;
  };

  it("accumulates something at every head count either model uses", () => {
    // AF3's trunk and template stack, then OpenDDE's.
    for (const [channels, heads, dimension, dialect] of [
      [128, 4, 32, ALPHAFOLD3], [64, 4, 16, ALPHAFOLD3],
      [384, 12, 32, OPENDDE], [64, 2, 32, OPENDDE],
    ]) {
      const source = biasFor(channels, heads, dimension, dialect);
      assert.match(source, /total0/,
        `${heads} heads of ${dimension} generated no accumulator`);
      // Every declared binding is referenced by the body, or WebGPU's inferred
      // layout will not match the bind group.
      const body = source.slice(source.indexOf("fn main"));
      for (const name of ["normalized", "projection", "bias"]) {
        assert.ok(body.includes(name), `${heads} heads: ${name} is never read`);
      }
    }
  });

  it("takes the scalar arm when the heads are not a multiple of four", () => {
    assert.match(biasFor(64, 2, 32, OPENDDE), /projection: array<f32>/);
    assert.match(biasFor(384, 12, 32, OPENDDE), /projection: array<vec4<f32>>/);
  });
});
